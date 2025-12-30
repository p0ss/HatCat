#!/usr/bin/env python3
"""
Test sunburst color API with a simple request.
"""

import requests
import json

def test_chat():
    url = "http://localhost:8765/v1/chat/completions"

    payload = {
        "model": "hatcat-divergence",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 20,
        "stream": True
    }

    print("=" * 80)
    print("TESTING SUNBURST COLOR API")
    print("=" * 80)
    print()
    print(f"Request: {payload['messages'][0]['content']}")
    print()

    response = requests.post(url, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk['choices'][0]['delta']

                    if 'content' in delta:
                        token = delta['content']
                        metadata = delta.get('metadata', {})

                        # Extract color info
                        color = metadata.get('color', '#808080')
                        palette = metadata.get('palette', [])
                        divergence = metadata.get('divergence', {})
                        max_div = divergence.get('max_divergence', 0.0)

                        # Display
                        print(f"Token: '{token}'")
                        print(f"  Color: {color}")
                        print(f"  Max Divergence: {max_div:.3f}")

                        if palette:
                            print(f"  Palette: {palette}")

                        # Show top divergences
                        top_divs = divergence.get('top_divergences', [])
                        if top_divs:
                            print(f"  Top Divergences:")
                            for d in top_divs:
                                print(f"    {d['concept']:30s} Δ={d['divergence']:.3f} (act:{d['activation']:.3f}, txt:{d['text']:.3f})")

                        print()

                except json.JSONDecodeError:
                    pass

if __name__ == "__main__":
    test_chat()
