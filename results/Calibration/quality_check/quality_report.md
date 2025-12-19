# Concept Pack Quality Report

**Generated**: 2025-12-06T10:59:33.701246

## Executive Summary

### Quadrant Distribution

| Quadrant | Description | Count | Est. F1 Ceiling |
|----------|-------------|-------|-----------------|
| A | Low synsets, Low siblings | 3501 | 0.88 |
| B | Low synsets, High siblings | 2703 | 0.83 |
| C | High synsets, High siblings | 549 | 0.76 |
| D | High synsets, Low siblings | 503 | 0.85 |

### Issues Detected

| Issue Type | Count | Description |
|------------|-------|-------------|
| polysemy | 1052 | High synset count - multiple word senses |
| dense_siblings | 165 | High child count - crowded negative space |
| combined | 549 | Both issues (Quadrant C) - critical |

### Severity Distribution

| Severity | Count |
|----------|-------|
| critical | 549 |
| high | 666 |
| medium | 551 |
| low | 0 |

## Top Concepts Needing Attention

| Concept | Layer | Quadrant | Synsets | Children | Issues | Est. F1 |
|---------|-------|----------|---------|----------|--------|---------|
| StationaryArtifact | 2 | C | 50 | 65 | polysemy, combined, dense_siblings | 0.76 |
| Sport | 2 | C | 50 | 37 | polysemy, combined, dense_siblings | 0.76 |
| BodyMotion | 2 | C | 50 | 34 | polysemy, combined, dense_siblings | 0.76 |
| EngineeringComponent | 2 | C | 50 | 34 | polysemy, combined, dense_siblings | 0.76 |
| BiologicallyActiveSubstance | 2 | C | 50 | 33 | polysemy, combined, dense_siblings | 0.76 |
| Translocation | 2 | C | 50 | 29 | polysemy, combined, dense_siblings | 0.76 |
| Medicine | 3 | C | 50 | 28 | polysemy, combined, dense_siblings | 0.76 |
| LandArea | 2 | C | 50 | 27 | polysemy, combined, dense_siblings | 0.76 |
| Mixture | 2 | C | 50 | 25 | polysemy, combined, dense_siblings | 0.76 |
| Organ | 2 | C | 50 | 24 | polysemy, combined, dense_siblings | 0.76 |
| Removing | 4 | C | 50 | 22 | polysemy, combined, dense_siblings | 0.76 |
| Transfer | 3 | C | 50 | 21 | polysemy, combined, dense_siblings | 0.76 |
| BodySubstance | 2 | C | 50 | 20 | polysemy, combined, dense_siblings | 0.76 |
| PreparedFood | 3 | C | 50 | 20 | polysemy, combined, dense_siblings | 0.76 |
| ContentDevelopment | 2 | C | 50 | 19 | polysemy, combined, dense_siblings | 0.76 |
| OrganizationalProcess | 2 | C | 50 | 19 | polysemy, combined, dense_siblings | 0.76 |
| Wine | 3 | C | 50 | 18 | polysemy, combined, dense_siblings | 0.76 |
| Room | 3 | C | 50 | 17 | polysemy, combined, dense_siblings | 0.76 |
| Bone | 2 | C | 50 | 15 | polysemy, combined, dense_siblings | 0.76 |
| Stating | 3 | C | 50 | 15 | polysemy, combined, dense_siblings | 0.76 |

## Recommended Remediations

Total meld operations generated: 1186

| Operation | Count |
|-----------|-------|
| split | 1186 |

### High Priority Actions

- **SPLIT** `Agreement`: Concept has 26 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `AIGrowth`: Concept has 23 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `Ambulating`: Concept has 10 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `AngleMeasure`: Concept has 25 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `AnimalShell`: Concept has 16 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `Article`: Concept has 15 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `ArtificialIntelligence`: Concept has 30 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `AttachingDevice`: Concept has 10 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `Beverage`: Concept has 50 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
- **SPLIT** `BiologicalAttribute`: Concept has 50 synsets indicating polysemy. Splitting into sense-specific lenses will improve activa...
