"""
title: HatCat Divergence Pipe
author: HatCat
version: 1.0.0
required_open_webui_version: 0.3.9
description: Proxy pipe that visualizes token-level concept divergence from HatCat server
"""

from typing import List, Union, Generator, Iterator
import requests
import json
import html


class Pipe:
    """
    OpenWebUI Pipe that connects to HatCat server and renders divergence visualizations.

    This pipe:
    1. Forwards chat requests to HatCat server
    2. Processes streaming responses with divergence metadata
    3. Renders tokens with sunburst colors and tooltips
    """

    def __init__(self):
        self.type = "manifold"
        self.name = "hatcat"

    def pipes(self) -> List[dict]:
        """Return list of available models from HatCat server"""
        return [
            {
                "id": "hatcat-divergence",
                "name": "HatCat Divergence Analyzer",
            }
        ]

    def pipe(self, body: dict = {}) -> Union[str, Generator, Iterator]:
        """
        Process chat completion request through HatCat server.

        Args:
            body: Request body from OpenWebUI containing messages, model, etc.

        Returns:
            Generator yielding plain text tokens (HTML won't render in OpenWebUI chat)
        """
        # Extract messages from body
        messages = body.get("messages", [])
        # Forward to HatCat server
        hatcat_url = "http://localhost:8765/v1/chat/completions"

        # Build request
        request_body = {
            "model": "hatcat-divergence",
            "messages": messages,
            "stream": True,
            "max_tokens": body.get("max_tokens", 512),
            "temperature": body.get("temperature", 0.7),
        }

        try:
            # Stream from HatCat server
            response = requests.post(
                hatcat_url,
                json=request_body,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # Skip "data: " prefix
                if line.startswith("data: "):
                    line = line[6:]

                # Check for end of stream
                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)

                    # Extract token and metadata
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})

                        content = delta.get("content", "")
                        metadata = delta.get("metadata", {})

                        if content:
                            # Add simple emoji indicator based on divergence level
                            decorated = self._decorate_token(content, metadata)
                            yield decorated

                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue

        except requests.exceptions.RequestException as e:
            yield f'<span style="color: red;">Error connecting to HatCat server: {str(e)}</span>'

    def _decorate_token(self, token: str, metadata: dict) -> str:
        """
        Add simple text decoration based on divergence (no HTML).

        Just returns plain token - visualization would need a browser extension.
        """
        # For now, just return plain token
        # OpenWebUI doesn't support HTML rendering in chat
        return token

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
            # No metadata - return plain token
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
            tooltip_lines.append(f"Max Divergence: {max_div:.3f}")

            top_divs = divergence.get("top_divergences", [])[:3]
            if top_divs:
                tooltip_lines.append("\\nTop Concepts:")
                for d in top_divs:
                    concept = d.get("concept", "")
                    act = d.get("activation", 0)
                    txt = d.get("text", 0)
                    div = d.get("divergence", 0)
                    tooltip_lines.append(
                        f"  • {concept}: act={act:.2f}, txt={txt:.2f}, div={div:.2f}"
                    )

        if palette:
            palette_str = ", ".join(palette[:5])
            tooltip_lines.append(f"\\nPalette: {palette_str}")

        if steering.get("active"):
            steerings = steering.get("steerings", [])
            if steerings:
                tooltip_lines.append("\\nActive Steerings:")
                for s in steerings:
                    concept = s.get("concept", "")
                    strength = s.get("strength", 0)
                    direction = "amplify" if strength > 0 else "suppress"
                    tooltip_lines.append(f"  • {concept}: {direction} ({strength:+.2f})")

        tooltip_content = "\\n".join(tooltip_lines)

        # Steering indicator emoji
        steering_indicator = "⚡" if steering.get("active") else ""

        # Render styled token with inline styles
        return f'''<span style="background-color: {color}; color: {text_color}; padding: 2px 4px; border-radius: 3px; cursor: help; display: inline-block; transition: all 0.2s ease;" title="{html.escape(tooltip_content)}">{steering_indicator}{html.escape(token)}</span>'''

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
