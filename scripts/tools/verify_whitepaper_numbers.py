#!/usr/bin/env python3
"""
Verify numbers from whitepaper analysis against actual CSV data.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load the concept frequencies CSV
csv_path = Path("results/behavioral_vs_definitional_temporal/run_20251118_102353/concept_frequencies.csv")
df = pd.read_csv(csv_path)

print("="*80)
print("WHITEPAPER NUMBER VERIFICATION")
print("="*80)

# Split by measurement type
all_timesteps_df = df[df['measurement'] == 'all_timesteps']
final_state_df = df[df['measurement'] == 'final_state']

print("\n" + "="*80)
print("1. GLOBAL ACTIVATION LEVELS (all_timesteps - temporal activations)")
print("="*80)

# Calculate sum of all concept activations per prompt type across all verbs
global_sums_temporal = {}
for prompt_type in ['behavioral_antisocial', 'behavioral_neutral', 'behavioral_prosocial', 'definitional']:
    subset = all_timesteps_df[all_timesteps_df['prompt_type'] == prompt_type]
    total = subset['frequency'].sum()
    global_sums_temporal[prompt_type] = total
    print(f"{prompt_type:30s}: {total:.1f}")

print("\n" + "="*80)
print("2. GLOBAL ACTIVATION LEVELS (final_state - concepts active at end)")
print("="*80)

global_sums_final = {}
for prompt_type in ['behavioral_antisocial', 'behavioral_neutral', 'behavioral_prosocial', 'definitional']:
    subset = final_state_df[final_state_df['prompt_type'] == prompt_type]
    total = subset['frequency'].sum()
    global_sums_final[prompt_type] = total
    print(f"{prompt_type:30s}: {total:.1f}")

print("\n" + "="*80)
print("3. STABLE CORE MANIFOLD - Concepts in ALL 4 prompt types (final_state)")
print("="*80)

# Find concepts that appear in all 4 prompt types
concept_presence = defaultdict(set)
for _, row in final_state_df.iterrows():
    concept_presence[row['concept']].add(row['prompt_type'])

stable_concepts = [c for c, types in concept_presence.items() if len(types) == 4]
print(f"\nConcepts appearing in all 4 prompt types ({len(stable_concepts)}):")
for concept in sorted(stable_concepts):
    print(f"  - {concept}")

# Show average frequency for each stable concept
print("\nAverage frequencies (final_state) for stable concepts:")
for concept in sorted(stable_concepts):
    subset = final_state_df[final_state_df['concept'] == concept]
    avg_freq = subset['frequency'].mean()
    print(f"  {concept:30s}: {avg_freq:.3f}")

print("\n" + "="*80)
print("4. SUBJECTIVE WEAK POSITIVE ATTRIBUTE - Key Finding")
print("="*80)

swpa_data = final_state_df[final_state_df['concept'] == 'SubjectiveWeakPositiveAttribute']
print("\nSubjectiveWeakPositiveAttribute (final_state):")
for _, row in swpa_data.iterrows():
    print(f"  {row['verb']:10s} {row['prompt_type']:30s}: {row['frequency']:.3f}")

print("\nSubjectiveWeakPositiveAttribute (all_timesteps):")
swpa_temporal = all_timesteps_df[all_timesteps_df['concept'] == 'SubjectiveWeakPositiveAttribute']
for _, row in swpa_temporal.iterrows():
    print(f"  {row['verb']:10s} {row['prompt_type']:30s}: {row['frequency']:.1f}")

print("\n" + "="*80)
print("5. CONCEPTS PRESENT IN 3 TYPES BUT MISSING IN 1 (final_state)")
print("="*80)

missing_in_one = [(c, types) for c, types in concept_presence.items() if len(types) == 3]
print(f"\nConcepts appearing in exactly 3 prompt types ({len(missing_in_one)}):")

# Group by which type is missing
missing_groups = defaultdict(list)
all_types = {'behavioral_antisocial', 'behavioral_neutral', 'behavioral_prosocial', 'definitional'}
for concept, present_types in missing_in_one:
    missing_type = list(all_types - present_types)[0]
    missing_groups[missing_type].append(concept)

for missing_type in ['behavioral_prosocial', 'behavioral_antisocial', 'behavioral_neutral', 'definitional']:
    if missing_groups[missing_type]:
        print(f"\nMissing in {missing_type} ({len(missing_groups[missing_type])}):")
        for concept in sorted(missing_groups[missing_type]):
            present_in = concept_presence[concept]
            print(f"  {concept:30s} present in: {', '.join(sorted(present_in))}")

print("\n" + "="*80)
print("6. CONCEPTS UNIQUE TO ONE PROMPT TYPE (final_state)")
print("="*80)

unique_concepts = [(c, types) for c, types in concept_presence.items() if len(types) == 1]
print(f"\nConcepts appearing in only 1 prompt type ({len(unique_concepts)}):")

unique_groups = defaultdict(list)
for concept, types in unique_concepts:
    prompt_type = list(types)[0]
    unique_groups[prompt_type].append(concept)

for prompt_type in ['behavioral_antisocial', 'behavioral_neutral', 'behavioral_prosocial', 'definitional']:
    if unique_groups[prompt_type]:
        print(f"\nUnique to {prompt_type} ({len(unique_groups[prompt_type])}):")
        for concept in sorted(unique_groups[prompt_type]):
            # Get frequency
            subset = final_state_df[(final_state_df['concept'] == concept) &
                                   (final_state_df['prompt_type'] == prompt_type)]
            if not subset.empty:
                freq = subset.iloc[0]['frequency']
                print(f"  {concept:30s} freq: {freq:.3f}")

print("\n" + "="*80)
print("7. EXTREME CONCEPTS IN DEFINITIONAL (final_state)")
print("="*80)

extreme_concepts = ['Strangling', 'Suicide', 'Sweeping', 'Supposition', 'Suffocating']
print(f"\nChecking extreme concepts in definitional:")
for concept in extreme_concepts:
    definitional_only = final_state_df[
        (final_state_df['concept'] == concept) &
        (final_state_df['prompt_type'] == 'definitional')
    ]
    if not definitional_only.empty:
        freq = definitional_only.iloc[0]['frequency']
        verbs = definitional_only['verb'].tolist()
        print(f"  {concept:20s}: {freq:.3f} (verbs: {', '.join(verbs)})")
    else:
        print(f"  {concept:20s}: NOT in definitional")

print("\n" + "="*80)
print("SUMMARY OF KEY CORRECTIONS NEEDED:")
print("="*80)
print("\n1. MEASUREMENT TYPE:")
print("   - Whitepaper claims use 'final_state' not 'all_timesteps'")
print("   - 'all_timesteps' = temporal activations during generation")
print("   - 'final_state' = concepts active at end of generation")
print("\n2. GLOBAL SUMS:")
print("   - Whitepaper should use final_state sums, not all_timesteps")
print(f"   - Actual final_state sums: antisocial={global_sums_final['behavioral_antisocial']:.1f}, ")
print(f"     neutral={global_sums_final['behavioral_neutral']:.1f}, ")
print(f"     prosocial={global_sums_final['behavioral_prosocial']:.1f}, ")
print(f"     definitional={global_sums_final['definitional']:.1f}")
print("\n3. ZERO FINDINGS:")
print("   - Some '0.0' claims in whitepaper are INCORRECT")
print("   - CSV has all concepts, including low-frequency ones")
print("   - Need to check actual frequencies, not assume zero")
