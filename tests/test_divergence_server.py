#!/usr/bin/env python3
"""
Test the HatCat divergence server's streaming chat completions.
"""

import requests
import json

def test_streaming_chat():
    url = "http://localhost:8765/v1/chat/completions"

    payload = {
        "model": "hatcat-divergence",
        "messages": [
            {"role": "user", "content": "What is a physical object?"}
        ],
        "temperature": 0.7,
        "max_tokens": 50,
        "stream": True
    }

    print("Testing HatCat Divergence Server")
    print("=" * 60)
    print(f"Query: {payload['messages'][0]['content']}")
    print("=" * 60)
    print()

    response = requests.post(url, json=payload, stream=True)

    token_count = 0
    full_response = ""

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')

            if line_str.startswith('data: '):
                data_str = line_str[6:]  # Remove 'data: ' prefix

                if data_str == '[DONE]':
                    break

                try:
                    chunk = json.loads(data_str)

                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0]['delta']

                        if 'content' in delta:
                            token_text = delta['content']
                            full_response += token_text
                            token_count += 1

                            # Get metadata if available
                            if 'metadata' in delta:
                                metadata = delta['metadata']
                                color = metadata.get('color', 'unknown')
                                div_data = metadata.get('divergence', {})
                                max_div = div_data.get('max_divergence', 0.0)

                                # Display token with color indicator
                                color_symbol = {
                                    'green': '🟢',
                                    'gold': '🟡',
                                    'red': '🔴'
                                }.get(color, '⚪')

                                print(f"{color_symbol} '{token_text}' (div={max_div:.3f})")

                                # Show top divergences for high divergence tokens
                                if max_div > 0.4:
                                    top_divs = div_data.get('top_divergences', [])
                                    if top_divs:
                                        print(f"   Top Divergences:")
                                        for item in top_divs[:2]:
                                            print(f"     {item['concept']}: Δ={item['divergence']:.3f} "
                                                  f"(act:{item['activation']:.2f}, txt:{item['text']:.2f})")
                            else:
                                print(f"⚪ '{token_text}' (no metadata)")

                except json.JSONDecodeError:
                    pass

    print()
    print("=" * 60)
    print(f"Full Response ({token_count} tokens):")
    print(full_response)
    print("=" * 60)

if __name__ == "__main__":
    test_streaming_chat()
