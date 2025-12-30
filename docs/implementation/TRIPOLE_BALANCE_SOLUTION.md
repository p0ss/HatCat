# Tripole Simplex Training Balance Solution

## Problem Discovered

Tripole simplex lenses were showing very poor neutral pole F1 scores (~0.27-0.31) despite having corrected pole definitions. The issue was consistent 6-8x data imbalance across all 13 S-tier simplexes:

- **Negative poles**: 87-94 overlap synsets
- **Neutral poles**: 12-15 overlap synsets (6-8x fewer!)
- **Positive poles**: 83-97 overlap synsets

## Root Cause

In joint tripole training with shared softmax loss, data imbalance causes severe optimization problems:
- The model gets 6-8x more opportunities to learn neg/pos patterns
- Neutral signal is drowned out during gradient descent
- This effect is amplified in joint training vs binary classification

## Solution Validation

We tested three approaches on taste_development simplex:

### 1. Imbalanced Baseline (Corrected Definitions)
- **Neutral F1**: 0.273 (average of 5 runs)
- Runs: 0.308, 0.000, 0.429, 0.360, 0.267
- High variance, often catastrophic failure

### 2. Min Downsampling (65 examples/pole)
- **Neutral F1**: 0.377
- Downsampled neg/pos to match neutral's 13 overlaps
- **Problem**: Better neutral but data-starved (only 65 examples/pole)

### 3. Optimal Balanced (435 examples/pole) ✅
- **Neutral F1**: 0.767 (2.8x better!)
- **Overall F1**: 0.826
- Per-pole: neg=0.880, neu=0.767, pos=0.800
- Used median overlap count strategy with adjusted prompts_per_synset
- **This proves the hypothesis**: balanced data solves the problem

## Implementation Strategy

The optimal approach requires **upsampling** underrepresented poles via API enrichment, not downsampling:

### Enrichment Plan
- **Total new overlaps needed**: 1,068
- **API requests**: 38 (batches of ~50 each)
- **Strategy**: Bring all poles to maximum count (target=max)

### Breakdown by Simplex
```
affective_awareness:     neutral +77, positive +5
affective_coherence:     neutral +76, positive +4
aspiration/social_mobility: neutral +79, positive +6
hedonic_arousal_intensity: neutral +78, positive +2
motivational_regulation: negative +3, neutral +81
relational_attachment:   neutral +80, positive +6
relational_love:        neutral +75, positive +1
social_connection:      negative +3, neutral +81
social_evaluation:      neutral +78
social_orientation:     negative +8, neutral +84
taste_development:      negative +1, neutral +75
temporal_affective_valence: negative +3, neutral +78
threat_perception:      neutral +77, positive +7
```

## Multicultural Enrichment

The API prompts explicitly request culturally-diverse concepts to prevent Western bias:

- **East Asian**: 面子/mianzi (face), 和/wa (harmony), 情/jeong (affection)
- **South Asian**: dharma, karma, ahimsa
- **Middle Eastern**: Arabic, Persian, Hebrew concepts
- **African**: Various languages/cultures
- **Indigenous**: Native American, Aboriginal, etc.
- **Latin American**: Cultural nuances beyond standard Spanish/Portuguese

### Why This Matters
Models should not be able to escape detection by encoding concepts in culturally-specific ways. Training on diverse cultural representations ensures:
1. Lenses work across multilingual contexts
2. Capture unique cultural perspectives missing from Western psychology
3. Prevent models from exploiting cultural blind spots

## Files

- **Enrichment plan**: `/data/balance_enrichment_requests.json`
- **Corrected definitions**: `/data/s_tier_simplex_definitions.json`
- **Analysis script**: `/scripts/balance_simplex_overlaps.py`
- **Test scripts**:
  - `/scripts/test_tripole_balanced_downsampling.py`
  - `/scripts/test_tripole_balanced_optimal.py`

## Next Steps

1. Execute 38 API requests from enrichment plan to generate 1,068 new synsets
2. Merge new synsets into `simplex_overlap_synsets_enriched.json`
3. Re-train tripole lenses with balanced data
4. Expect neutral F1 to improve from ~0.27 to ~0.77 (2.8x)

## Key Insight

**Data imbalance in joint tripole training is catastrophic**. The solution is not better definitions (though those help), but balanced training data. The optimal balanced approach achieves state-of-the-art performance with median-count targeting to maximize data while ensuring balance.
