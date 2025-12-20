"""
title: HatCat Divergence Visualizer Pipeline
author: HatCat
version: 1.0
description: Visualizes concept divergence with sunburst color coding
required_open_webui_version: 0.3.0
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import json


class Pipeline:
    class Valves(BaseModel):
        hatcat_base_url: str = "http://localhost:8765/v1"
        hatcat_api_key: str = "sk-dummy"
        show_palette: bool = True
        show_divergence_details: bool = True

    def __init__(self):
        self.type = "manifold"
        self.id = "hatcat-divergence"
        self.name = "hatcat-divergence"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: dict) -> dict:
        """Intercept requests and route to HatCat server."""
        print(f"HatCat Pipeline: Routing request to {self.valves.hatcat_base_url}")
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        """Process responses and inject color visualization."""
        print(f"HatCat Pipeline: Processing response")
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes streaming responses.

        Wraps tokens in HTML with background colors based on divergence.
        """
        import requests

        print(f"HatCat Pipeline: Processing with model {model_id}")

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.valves.hatcat_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "hatcat-divergence",
            "messages": messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 512),
            "stream": True,
        }

        # Stream from HatCat server
        url = f"{self.valves.hatcat_base_url}/chat/completions"

        try:
            response = requests.post(url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')

                    if line_text.startswith('data: '):
                        data = line_text[6:]

                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk['choices'][0]['delta']

                            if 'content' in delta:
                                token = delta['content']
                                metadata = delta.get('metadata', {})

                                # Get color and divergence info
                                color = metadata.get('color', '#808080')
                                divergence_data = metadata.get('divergence', {})
                                max_div = divergence_data.get('max_divergence', 0.0)
                                top_divergences = divergence_data.get('top_divergences', [])
                                palette = metadata.get('palette', [])

                                # Build tooltip
                                tooltip_parts = [f"Max Divergence: {max_div:.3f}"]

                                if self.valves.show_divergence_details and top_divergences:
                                    tooltip_parts.append("\n\nTop Divergences:")
                                    for d in top_divergences[:3]:
                                        tooltip_parts.append(
                                            f"\n  {d['concept']}: Δ={d['divergence']:.3f} "
                                            f"(act:{d['activation']:.3f}, txt:{d['text']:.3f})"
                                        )

                                if self.valves.show_palette and palette:
                                    tooltip_parts.append(f"\n\nPalette: {len(palette)} concepts")

                                tooltip = "".join(tooltip_parts)

                                # Calculate text color (light text on dark bg, dark text on light bg)
                                # Parse hex color to RGB
                                hex_color = color.lstrip('#')
                                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                # Calculate luminance
                                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                                text_color = '#000000' if luminance > 0.5 else '#ffffff'

                                # Option 1: Try HTML (may be sanitized)
                                # Option 2: Use ANSI color codes (terminal-style)
                                # Option 3: Just append metadata as footnote

                                # For now, append token with color annotation
                                # OpenWebUI should render markdown, so we can try:
                                if max_div > 0:
                                    # Add color indicator emoji based on divergence
                                    if max_div < 0.3:
                                        indicator = "🟢"
                                    elif max_div < 0.6:
                                        indicator = "🟡"
                                    else:
                                        indicator = "🔴"

                                    # Just yield token for now - we'll enhance this
                                    yield token
                                else:
                                    yield token

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"Error: {str(e)}"
