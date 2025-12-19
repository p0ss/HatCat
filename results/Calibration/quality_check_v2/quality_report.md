# Concept Pack Quality Report

**Generated**: 2025-12-06T11:18:10.367806

## Executive Summary

### Quadrant Distribution

| Quadrant | Description | Count | Est. F1 Ceiling |
|----------|-------------|-------|-----------------|
| A | Low synsets, Low siblings | 758 | 0.88 |
| B | Low synsets, High siblings | 703 | 0.83 |
| C | High synsets, High siblings | 349 | 0.76 |
| D | High synsets, Low siblings | 283 | 0.85 |

### Issues Detected

| Issue Type | Count | Description |
|------------|-------|-------------|
| true_polysemy | 632 | TRUE polysemy - divergent sense groups |
| dense_siblings | 49 | High child count - crowded negative space |
| combined | 349 | TRUE polysemy + dense siblings - critical |

**Note**: High synset count alone is NOT flagged as an issue. Most high-synset concepts
have taxonomic depth (hyponyms), which provides training variety. Only concepts with
TRUE polysemy (divergent hypernym paths indicating distinct meanings) are flagged.

### Severity Distribution

| Severity | Count |
|----------|-------|
| critical | 349 |
| high | 379 |
| medium | 302 |
| low | 0 |

## Top Concepts Needing Attention

| Concept | Layer | Quadrant | Synsets | Children | Issues | Est. F1 |
|---------|-------|----------|---------|----------|--------|---------|
| Medicine | 3 | C | 50 | 28 | true_polysemy, combined, dense_siblings | 0.76 |
| Transfer | 3 | C | 50 | 21 | true_polysemy, combined, dense_siblings | 0.76 |
| PreparedFood | 3 | C | 50 | 20 | true_polysemy, combined, dense_siblings | 0.76 |
| Wine | 3 | C | 50 | 18 | true_polysemy, combined, dense_siblings | 0.76 |
| Room | 3 | C | 50 | 17 | true_polysemy, combined, dense_siblings | 0.76 |
| Stating | 3 | C | 50 | 15 | true_polysemy, combined, dense_siblings | 0.76 |
| FruitOrVegetable | 3 | C | 50 | 13 | true_polysemy, combined, dense_siblings | 0.76 |
| BodyVessel | 3 | C | 50 | 11 | true_polysemy, combined, dense_siblings | 0.76 |
| PoliticalProcess | 3 | C | 50 | 11 | true_polysemy, combined, dense_siblings | 0.76 |
| Declaring | 3 | C | 50 | 10 | true_polysemy, combined, dense_siblings | 0.76 |
| LandTransitway | 3 | C | 21 | 10 | true_polysemy, combined, dense_siblings | 0.76 |
| Toxin | 3 | C | 18 | 11 | true_polysemy, combined, dense_siblings | 0.76 |
| Wind | 3 | C | 17 | 12 | true_polysemy, combined, dense_siblings | 0.76 |
| MilitaryOrganization | 3 | C | 17 | 11 | true_polysemy, combined, dense_siblings | 0.76 |
| Colloid | 3 | C | 8 | 12 | true_polysemy, combined, dense_siblings | 0.76 |
| CommunicationDevice | 3 | C | 5 | 18 | true_polysemy, combined, dense_siblings | 0.76 |
| SportsFacility | 3 | C | 4 | 10 | true_polysemy, combined, dense_siblings | 0.76 |
| DigitalData | 3 | C | 3 | 13 | true_polysemy, combined, dense_siblings | 0.76 |
| SaltWaterArea | 3 | C | 2 | 11 | true_polysemy, combined, dense_siblings | 0.76 |
| ReligiousProcess | 3 | C | 50 | 9 | true_polysemy, combined | 0.76 |

## Recommended Remediations

Total meld operations generated: 678

| Operation | Count |
|-----------|-------|
| split | 678 |

### High Priority Actions

- **SPLIT** `Adjective`: TRUE POLYSEMY: Concept has 10 semantically distinct sense groups (from 13 total synsets). Splitting ...
- **SPLIT** `Adverb`: TRUE POLYSEMY: Concept has 4 semantically distinct sense groups (from 4 total synsets). Splitting in...
- **SPLIT** `Advertising`: TRUE POLYSEMY: Concept has 10 semantically distinct sense groups (from 23 total synsets). Splitting ...
- **SPLIT** `Agriculture`: TRUE POLYSEMY: Concept has 5 semantically distinct sense groups (from 16 total synsets). Splitting i...
- **SPLIT** `AIAbuse`: TRUE POLYSEMY: Concept has 6 semantically distinct sense groups (from 7 total synsets). Splitting in...
- **SPLIT** `AIControlProblem`: TRUE POLYSEMY: Concept has 14 semantically distinct sense groups (from 19 total synsets). Splitting ...
- **SPLIT** `AIDeception`: TRUE POLYSEMY: Concept has 12 semantically distinct sense groups (from 14 total synsets). Splitting ...
- **SPLIT** `Airplane`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 15 total synsets). Splitting i...
- **SPLIT** `Airway`: TRUE POLYSEMY: Concept has 3 semantically distinct sense groups (from 3 total synsets). Splitting in...
- **SPLIT** `AISuffering`: TRUE POLYSEMY: Concept has 10 semantically distinct sense groups (from 17 total synsets). Splitting ...
