#!/usr/bin/env python3
"""Test Layer 0 concept detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.hat import extract_concept_vector

# Load model
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-pt')
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-4b-pt',
    torch_dtype=torch.float16,
    device_map='cuda'
)
print('✓ Model loaded\n')

# Test detection for Layer 0 concepts
concepts = ['Physical', 'Quantity', 'Proposition', 'Entity', 'Process',
            'Object', 'Attribute', 'Relation', 'List', 'Collection', 'Abstract']

print('Testing concept detection...\n')
print('='*60)

for concept in concepts[:3]:  # Test first 3 concepts
    print(f'\nConcept: {concept}')
    print('-'*60)

    # Generate definition
    prompt = f'Define the concept "{concept}":'
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Generated: {text[len(prompt):].strip()[:80]}...')

    # Try to extract concept vector (this will fail if classifier not trained)
    try:
        v = extract_concept_vector(model, tokenizer, concept, device='cuda')
        print(f'✓ Concept vector extracted: shape {v.shape}')
        print(f'  Vector norm: {torch.norm(v).item():.4f}')
    except Exception as e:
        print(f'✗ Failed to extract: {e}')

print('\n' + '='*60)
print('Detection test complete')
