#!/usr/bin/env python3
"""
Visualize concept sunburst positions as an interactive HTML chart.

Creates an SVG sunburst diagram showing:
- Concentric rings for each layer
- Angular positions determining hue
- Color coding showing each concept's assigned color
- Hover tooltips with concept details
"""

import json
from pathlib import Path
import colorsys

def hsl_to_rgb(h, s, l):
    """Convert HSL to RGB (0-255)."""
    r, g, b = colorsys.hls_to_rgb(h/360.0, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

def generate_sunburst_html(positions: dict, output_file: Path):
    """Generate interactive HTML sunburst visualization."""

    # Group by layer
    by_layer = {}
    for concept, pos in positions.items():
        layer = pos['layer']
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append((concept, pos))

    max_layer = max(by_layer.keys())

    # Generate SVG
    width = 1200
    height = 1200
    center_x = width / 2
    center_y = height / 2
    max_radius = min(width, height) / 2 - 50

    svg_elements = []

    # Draw each layer
    for layer in sorted(by_layer.keys()):
        concepts = by_layer[layer]

        # Ring dimensions
        inner_radius = (layer / (max_layer + 1)) * max_radius
        outer_radius = ((layer + 1) / (max_layer + 1)) * max_radius

        for concept, pos in concepts:
            angle = pos['angle']
            hue = angle
            saturation = 0.7
            lightness = 0.6  # Medium lightness for reference chart

            r, g, b = hsl_to_rgb(hue, saturation, lightness)
            color = f"rgb({r},{g},{b})"

            # For simplicity, draw as a thin wedge
            # Calculate angular span (assume equal division among siblings)
            num_concepts = len(concepts)
            angular_span = 360.0 / num_concepts if num_concepts > 0 else 360.0

            # Arc path
            start_angle = angle - angular_span / 2
            end_angle = angle + angular_span / 2

            # Convert to radians
            import math
            start_rad = math.radians(start_angle - 90)  # -90 to start at top
            end_rad = math.radians(end_angle - 90)

            # Calculate arc points
            x1_inner = center_x + inner_radius * math.cos(start_rad)
            y1_inner = center_y + inner_radius * math.sin(start_rad)
            x1_outer = center_x + outer_radius * math.cos(start_rad)
            y1_outer = center_y + outer_radius * math.sin(start_rad)

            x2_inner = center_x + inner_radius * math.cos(end_rad)
            y2_inner = center_y + inner_radius * math.sin(end_rad)
            x2_outer = center_x + outer_radius * math.cos(end_rad)
            y2_outer = center_y + outer_radius * math.sin(end_rad)

            # Large arc flag
            large_arc = 1 if angular_span > 180 else 0

            # SVG path
            path = f"""
            M {x1_inner},{y1_inner}
            L {x1_outer},{y1_outer}
            A {outer_radius},{outer_radius} 0 {large_arc},1 {x2_outer},{y2_outer}
            L {x2_inner},{y2_inner}
            A {inner_radius},{inner_radius} 0 {large_arc},0 {x1_inner},{y1_inner}
            Z
            """

            # Tooltip info
            tooltip = f"{concept}\\nLayer: {layer}\\nAngle: {angle:.1f}°\\nHue: {int(hue)}°\\nChildren: {len(pos.get('children', []))}"

            svg_elements.append(f'''
            <path d="{path}" fill="{color}" stroke="white" stroke-width="0.5" opacity="0.9">
                <title>{tooltip}</title>
            </path>
            ''')

    # Generate HTML
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HatCat Concept Sunburst</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #ffffff;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .chart-container {{
            display: flex;
            gap: 40px;
            align-items: start;
        }}
        svg {{
            background: #0a0a0a;
            border-radius: 8px;
        }}
        .legend {{
            flex-shrink: 0;
            width: 300px;
        }}
        .legend-item {{
            margin-bottom: 20px;
        }}
        .legend-item h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #aaa;
        }}
        .color-bar {{
            height: 30px;
            border-radius: 4px;
            margin-bottom: 5px;
        }}
        .gradient-bar {{
            background: linear-gradient(to right,
                hsl(0, 70%, 60%),
                hsl(60, 70%, 60%),
                hsl(120, 70%, 60%),
                hsl(180, 70%, 60%),
                hsl(240, 70%, 60%),
                hsl(300, 70%, 60%),
                hsl(360, 70%, 60%)
            );
        }}
        .brightness-bar {{
            background: linear-gradient(to right,
                hsl(200, 70%, 90%),
                hsl(200, 70%, 50%),
                hsl(200, 70%, 10%)
            );
        }}
        .legend-label {{
            font-size: 12px;
            color: #666;
            display: flex;
            justify-content: space-between;
        }}
        .layer-info {{
            background: #222;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
        .layer-info h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
        }}
        .layer-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #333;
            font-size: 12px;
        }}
        .layer-row:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎩 HatCat Concept Sunburst</h1>
        <div class="subtitle">
            Color mapping for divergence visualization<br>
            Hue = Angular Position | Brightness = Divergence Level
        </div>

        <div class="chart-container">
            <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
                {''.join(svg_elements)}

                <!-- Center label -->
                <text x="{center_x}" y="{center_y}" text-anchor="middle" font-size="24" fill="#666">
                    SUMO
                </text>
                <text x="{center_x}" y="{center_y + 25}" text-anchor="middle" font-size="14" fill="#444">
                    Ontology
                </text>
            </svg>

            <div class="legend">
                <div class="legend-item">
                    <h3>HUE: Concept Position</h3>
                    <div class="color-bar gradient-bar"></div>
                    <div class="legend-label">
                        <span>0° (Red)</span>
                        <span>180° (Cyan)</span>
                        <span>360° (Red)</span>
                    </div>
                    <p style="font-size: 12px; color: #888; margin-top: 10px;">
                        Each concept's angular position in the sunburst determines its hue on the color wheel.
                        Related concepts cluster together.
                    </p>
                </div>

                <div class="legend-item">
                    <h3>BRIGHTNESS: Divergence</h3>
                    <div class="color-bar brightness-bar"></div>
                    <div class="legend-label">
                        <span>Low (0.0)</span>
                        <span>Medium (0.5)</span>
                        <span>High (1.0)</span>
                    </div>
                    <p style="font-size: 12px; color: #888; margin-top: 10px;">
                        Brightness decreases as divergence increases.
                        Bright = model's thoughts align with text.
                        Dark = high divergence between thinking and writing.
                    </p>
                </div>

                <div class="layer-info">
                    <h3>Layer Distribution</h3>
                    {"".join([f'''
                    <div class="layer-row">
                        <span>Layer {layer}</span>
                        <span>{len(concepts)} concepts</span>
                    </div>
                    ''' for layer, concepts in sorted(by_layer.items())])}
                </div>

                <div style="margin-top: 20px; font-size: 12px; color: #666;">
                    <p><strong>How to use:</strong></p>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>Hover over segments to see concept details</li>
                        <li>Inner rings = higher-level concepts</li>
                        <li>Outer rings = more specific concepts</li>
                        <li>Similar hues = semantically related</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    '''

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)

def main():
    print("=" * 80)
    print("VISUALIZING CONCEPT SUNBURST")
    print("=" * 80)
    print()

    positions_file = Path('results/concept_sunburst_positions.json')

    if not positions_file.exists():
        print(f"Error: {positions_file} not found")
        print("Run: poetry run python scripts/build_concept_sunburst_positions_simple.py")
        return

    print("Loading positions...")
    with open(positions_file) as f:
        positions = json.load(f)

    print(f"✓ Loaded {len(positions)} concept positions")
    print()

    output_file = Path('results/concept_sunburst_visualization.html')
    print("Generating HTML visualization...")
    generate_sunburst_html(positions, output_file)
    print(f"✓ Saved to: {output_file}")
    print()

    print("Open in browser:")
    print(f"  file://{output_file.absolute()}")
    print()

if __name__ == "__main__":
    main()
