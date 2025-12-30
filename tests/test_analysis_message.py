#!/usr/bin/env python3
"""Test that HatCat sends analysis messages."""

import requests
import json

url = "http://localhost:8765/v1/chat/completions"

payload = {
    "model": "hatcat-divergence",
    "messages": [
        {"role": "user", "content": "What is AI?"}
    ],
    "stream": True,
    "max_tokens": 20,
    "temperature": 0.7,
    "session_id": "test-session"
}

print("Sending request...")
response = requests.post(url, json=payload, stream=True)

messages = []
analysis_found = False

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                model = chunk.get('model', '')
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')

                if model == 'hatcat-analyzer':
                    print(f"\n🎩 ANALYSIS MESSAGE FOUND!")
                    print(f"   Content: {content}")
                    analysis_found = True
                elif content:
                    print(content, end='', flush=True)

            except json.JSONDecodeError:
                pass

if analysis_found:
    print("\n\n✅ SUCCESS: Analysis message was sent from hatcat-analyzer model!")
else:
    print("\n\n❌ FAILED: No analysis message found")
