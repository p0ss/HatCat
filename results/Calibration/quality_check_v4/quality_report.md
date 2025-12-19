# Concept Pack Quality Report

**Generated**: 2025-12-06T11:36:39.342430

## Executive Summary

### Quadrant Distribution

| Quadrant | Description | Count | Est. F1 Ceiling |
|----------|-------------|-------|-----------------|
| A | Low synsets, Low siblings | 958 | 0.88 |
| B | Low synsets, High siblings | 954 | 0.83 |
| C | High synsets, High siblings | 98 | 0.76 |
| D | High synsets, Low siblings | 83 | 0.85 |

### Issues Detected

| Issue Type | Count | Description |
|------------|-------|-------------|
| true_polysemy | 181 | TRUE polysemy - divergent sense groups |
| dense_siblings | 49 | High child count - crowded negative space |
| combined | 114 | TRUE polysemy + dense siblings - critical |

**Note**: High synset count alone is NOT flagged as an issue. Most high-synset concepts
have taxonomic depth (hyponyms), which provides training variety. Only concepts with
TRUE polysemy (divergent hypernym paths indicating distinct meanings) are flagged.

### Severity Distribution

| Severity | Count |
|----------|-------|
| critical | 114 |
| high | 25 |
| medium | 205 |
| low | 0 |

## Top Concepts Needing Attention

| Concept | Layer | Quadrant | Synsets | Children | Issues | Est. F1 |
|---------|-------|----------|---------|----------|--------|---------|
| PreparedFood | 3 | C | 50 | 20 | combined, dense_siblings, true_polysemy | 0.76 |
| Stating | 3 | C | 50 | 15 | combined, dense_siblings, true_polysemy | 0.76 |
| FruitOrVegetable | 3 | C | 50 | 13 | combined, dense_siblings, true_polysemy | 0.76 |
| PoliticalProcess | 3 | C | 50 | 11 | combined, dense_siblings, true_polysemy | 0.76 |
| Declaring | 3 | C | 50 | 10 | combined, dense_siblings, true_polysemy | 0.76 |
| LandTransitway | 3 | C | 21 | 10 | combined, dense_siblings, true_polysemy | 0.76 |
| Wind | 3 | C | 17 | 12 | combined, dense_siblings, true_polysemy | 0.76 |
| MilitaryOrganization | 3 | C | 17 | 11 | combined, dense_siblings, true_polysemy | 0.76 |
| Colloid | 3 | C | 8 | 12 | combined, dense_siblings, true_polysemy | 0.76 |
| SportsFacility | 3 | C | 4 | 10 | combined, dense_siblings, true_polysemy | 0.76 |
| SaltWaterArea | 3 | C | 2 | 11 | combined, dense_siblings, true_polysemy | 0.76 |
| Transfer | 3 | B | 50 | 21 | combined, dense_siblings | 0.83 |
| Tissue | 3 | C | 50 | 8 | combined, true_polysemy | 0.76 |
| Cooking | 3 | C | 50 | 6 | combined, true_polysemy | 0.76 |
| Hormone | 3 | C | 50 | 5 | combined, true_polysemy | 0.76 |
| RegulatoryProcess | 3 | C | 50 | 4 | combined, true_polysemy | 0.76 |
| City | 3 | C | 50 | 3 | combined, true_polysemy | 0.76 |
| Nation | 3 | C | 50 | 3 | combined, true_polysemy | 0.76 |
| Character | 3 | C | 50 | 2 | combined, true_polysemy | 0.76 |
| Hair | 3 | C | 50 | 1 | combined, true_polysemy | 0.76 |

## Recommended Remediations

Total meld operations generated: 227

| Operation | Count |
|-----------|-------|
| split | 227 |

### High Priority Actions

- **SPLIT** `Airway`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 3 total synsets). Splitting in...
- **SPLIT** `Alga`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 50 total synsets). Splitting i...
- **SPLIT** `AtmosphericRegion`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 12 total synsets). Splitting i...
- **SPLIT** `Canine`: TRUE POLYSEMY: Concept has 4 semantically distinct sense groups (from 50 total synsets). Splitting i...
- **SPLIT** `City`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 50 total synsets). Splitting i...
- **SPLIT** `DramaticActing`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 7 total synsets). Splitting in...
- **SPLIT** `Feline`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 41 total synsets). Splitting i...
- **SPLIT** `Fern`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 50 total synsets). Splitting i...
- **SPLIT** `FreshWaterArea`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 5 total synsets). Splitting in...
- **SPLIT** `GameShot`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 10 total synsets). Splitting i...
