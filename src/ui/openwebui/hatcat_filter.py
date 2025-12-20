"""
title: HatCat Divergence Visualizer
author: HatCat
version: 1.0.0
description: Visualizes token-level concept divergence with sunburst colors and interactive steering
"""

from typing import Optional, Callable, Awaitable
import json
import html


class Filter:
    """
    OpenWebUI Filter that renders HatCat divergence metadata as colored tokens.

    This filter intercepts streaming responses from the HatCat server and:
    1. Extracts divergence metadata from delta.metadata
    2. Renders tokens with sunburst colors
    3. Adds hover tooltips with divergence details
    4. Provides steering controls via right-click menus
    """

    def __init__(self):
        self.valves = self.Valves()

    class Valves:
        """Configuration for the filter"""
        hatcat_api_base: str = "http://localhost:8765/v1"
        enable_tooltips: bool = True
        enable_steering: bool = True

    async def on_start(self, __user__: dict, __event_emitter__=None) -> None:
        """Initialize the filter when conversation starts"""
        # Inject CSS and JavaScript for visualization
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "🎩 HatCat Divergence Visualizer active",
                    "done": False
                }
            })

            # Inject CSS for token styling
            css = """
            <style>
            .hatcat-token {
                padding: 2px 4px;
                border-radius: 3px;
                cursor: help;
                position: relative;
                display: inline-block;
                transition: all 0.2s ease;
            }
            .hatcat-token:hover {
                transform: scale(1.05);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .hatcat-tooltip {
                position: absolute;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 10000;
                max-width: 350px;
                pointer-events: none;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            .hatcat-palette {
                display: flex;
                gap: 4px;
                margin-top: 8px;
            }
            .hatcat-palette-swatch {
                width: 16px;
                height: 16px;
                border-radius: 2px;
            }
            .steering-indicator {
                position: absolute;
                top: -4px;
                right: -4px;
                font-size: 10px;
            }
            </style>
            """
            await __event_emitter__({
                "type": "message",
                "data": {"content": css}
            })

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> dict:
        """
        Process streaming response and render colored tokens with divergence metadata.

        This is called for each chunk in the streaming response.
        """

        # Process messages in the response
        if "messages" in body:
            for message in body["messages"]:
                if message.get("role") == "assistant" and "content" in message:
                    content = message["content"]

                    # If content is a dict with metadata, render it
                    if isinstance(content, dict) and "metadata" in content:
                        token_text = content.get("text", "")
                        metadata = content.get("metadata", {})

                        # Render token with styling
                        message["content"] = self._render_token(token_text, metadata)
                    elif isinstance(content, str):
                        # Plain text - check if we can extract metadata from elsewhere
                        pass

        return body

    def _render_token(self, token: str, metadata: dict) -> str:
        """
        Render a single token with divergence visualization.

        Args:
            token: The text token
            metadata: Divergence metadata including color, palette, etc.

        Returns:
            HTML string with styled token
        """
        if not metadata or "divergence" not in metadata:
            return html.escape(token)

        color = metadata.get("color", "#808080")
        divergence = metadata.get("divergence", {})
        palette = metadata.get("palette", [])
        steering = metadata.get("steering", {})

        # Calculate contrasting text color
        luminance = self._get_luminance(color)
        text_color = "#000000" if luminance > 0.5 else "#ffffff"

        # Build tooltip content
        tooltip_lines = []

        max_div = divergence.get("max_divergence", 0)
        if max_div > 0:
            tooltip_lines.append(f"<b>Max Divergence:</b> {max_div:.3f}")

            top_divs = divergence.get("top_divergences", [])[:3]
            if top_divs:
                tooltip_lines.append("<br><b>Top Concepts:</b>")
                for d in top_divs:
                    concept = d.get("concept", "")
                    act = d.get("activation", 0)
                    txt = d.get("text", 0)
                    div = d.get("divergence", 0)
                    tooltip_lines.append(
                        f"  • {concept}: act={act:.2f}, txt={txt:.2f}, div={div:.2f}"
                    )

        if palette:
            palette_html = "".join(
                f'<span class="hatcat-palette-swatch" style="background-color: {c}"></span>'
                for c in palette[:5]
            )
            tooltip_lines.append(
                f'<br><b>Palette:</b><div class="hatcat-palette">{palette_html}</div>'
            )

        if steering.get("active"):
            steerings = steering.get("steerings", [])
            if steerings:
                tooltip_lines.append("<br><b>Active Steerings:</b>")
                for s in steerings:
                    concept = s.get("concept", "")
                    strength = s.get("strength", 0)
                    direction = "amplify" if strength > 0 else "suppress"
                    tooltip_lines.append(f"  • {concept}: {direction} ({strength:+.2f})")

        tooltip_content = "<br>".join(tooltip_lines)

        # Steering indicator emoji
        steering_indicator = "⚡" if steering.get("active") else ""

        # Render styled token
        return f'''<span class="hatcat-token"
            style="background-color: {color}; color: {text_color};"
            title="{html.escape(tooltip_content)}"
            data-token="{html.escape(token)}">
            {steering_indicator}{html.escape(token)}
        </span>'''

    def _get_luminance(self, hex_color: str) -> float:
        """Calculate relative luminance (WCAG formula)"""
        hex_color = hex_color.replace("#", "")
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255

        def adjust(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = adjust(r), adjust(g), adjust(b)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
