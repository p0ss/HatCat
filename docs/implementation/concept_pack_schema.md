# Concept Pack Schema

A **concept pack** is a distributable unit containing:
- Base ontology definitions
- Relationship mappings (e.g., WordNet)
- Hierarchy stitching logic
- Domain extensions
- Trained lenses for specific models

## Schema v1.0

```json
{
  "pack_id": "sumo-wordnet-aisafety-v1",
  "version": "1.0.0",
  "created": "2025-11-08T12:00:00Z",
  "description": "SUMO ontology with WordNet relationships and AISafety domain extension",

  "ontology_stack": {
    "base_ontology": {
      "name": "SUMO",
      "version": "2003",
      "source": "http://www.ontologyportal.org/",
      "concepts_file": "data/concept_graph/sumo_source/sumo_concepts.txt"
    },

    "relationship_sources": [
      {
        "name": "WordNet",
        "version": "3.0",
        "description": "WordNet 3.0 synsets and hyponym relationships"
      },
      {
        "name": "SUMO-WordNet Mappings",
        "version": "2.0",
        "mapping_file": "data/concept_graph/sumo_wordnet_mappings.json",
        "description": "Maps SUMO concepts to WordNet synsets"
      }
    ],

    "hierarchy_builder": {
      "script": "src/build_sumo_wordnet_layers.py",
      "version": "2.0",
      "method": "hyponym-based layering with adaptive depth",
      "parameters": {
        "max_layers": 6,
        "min_synsets_per_concept": 50,
        "hyponym_weighting": true
      }
    },

    "domain_extensions": [
      {
        "name": "AISafety",
        "version": "1.0",
        "concepts_file": "data/concept_graph/domains/aisafety/concepts.json",
        "parent_mappings_file": "data/concept_graph/domains/aisafety/parent_mappings.json",
        "description": "AI safety domain concepts (deception, alignment, etc.)"
      }
    ]
  },

  "concept_metadata": {
    "total_concepts": 5582,
    "layers": [0, 1, 2, 3, 4, 5],
    "layer_distribution": {
      "0": 14,
      "1": 120,
      "2": 850,
      "3": 2100,
      "4": 1800,
      "5": 698
    }
  },

  "model_lenses": [
    {
      "model_id": "gemma-3-4b-pt",
      "model_name": "google/gemma-3-4b-pt",
      "training_date": "2025-11-08",
      "layers": [0, 1, 2, 3, 4, 5],
      "lens_types": ["activation", "text"],
      "training_method": "dual_adaptive",
      "lens_paths": {
        "activation": "results/models/gemma-3-4b-pt/packs/sumo-wordnet-aisafety-v1/lenses/activation",
        "text": "results/models/gemma-3-4b-pt/packs/sumo-wordnet-aisafety-v1/lenses/text"
      },
      "concept_positions_file": "results/models/gemma-3-4b-pt/packs/sumo-wordnet-aisafety-v1/concept_sunburst_positions.json",
      "training_results": {
        "avg_f1_activation": 0.985,
        "avg_f1_text": 0.923,
        "graduation_rate": 0.98
      }
    }
  ],

  "compatibility": {
    "hatcat_version": "0.1.0",
    "required_dependencies": {
      "wordnet": "3.0",
      "nltk": ">=3.8"
    }
  },

  "distribution": {
    "license": "MIT",
    "authors": ["HatCat Team"],
    "repository": "https://github.com/yourname/hatcat",
    "download_url": "https://huggingface.co/hatcat/sumo-wordnet-aisafety-v1"
  }
}
```

## File Structure

```
concept_packs/
  sumo-wordnet-aisafety-v1/
    pack.json                          # Main metadata file (schema above)
    ontology/
      sumo_concepts.txt                # Base SUMO definitions
      wordnet_mappings.json            # WordNet → SUMO mappings
      aisafety_extension.json          # Domain concepts
    hierarchy/
      build_config.json                # Hierarchy builder parameters
      layer_assignments.json           # Concept → layer mappings
    models/
      gemma-3-4b-pt/
        lenses/
          activation/                   # Activation lenses
          text/                         # Text lenses
        concept_sunburst_positions.json
        training_results.json
```

## API Changes

### 1. List available concept packs
```
GET /v1/concept-packs
Response:
{
  "packs": [
    {
      "pack_id": "sumo-wordnet-aisafety-v1",
      "version": "1.0.0",
      "total_concepts": 5582,
      "supported_models": ["gemma-3-4b-pt", "llama-3-8b"]
    }
  ]
}
```

### 2. Get pack details
```
GET /v1/concept-packs/{pack_id}
Response: (full pack.json schema)
```

### 3. List models with pack support
```
GET /v1/models
Response:
{
  "models": [
    {
      "model_id": "gemma-3-4b-pt",
      "model_name": "google/gemma-3-4b-pt",
      "available_packs": [
        {
          "pack_id": "sumo-wordnet-aisafety-v1",
          "total_concepts": 5582,
          "lens_types": ["activation", "text"]
        }
      ]
    }
  ]
}
```

### 4. Chat with specific model + pack
```
POST /v1/chat/completions
{
  "model": "gemma-3-4b-pt",
  "concept_pack": "sumo-wordnet-aisafety-v1",  # NEW FIELD
  "messages": [...]
}
```

## Distribution Format

Concept packs can be distributed as:
1. **Git repos** (development)
2. **HuggingFace datasets** (community sharing)
3. **Zip archives** (simple download)

Example HuggingFace dataset structure:
```
hatcat/sumo-wordnet-aisafety-v1
├── README.md
├── pack.json
├── ontology/
└── models/
    ├── gemma-3-4b-pt/
    └── llama-3-8b/
```

## Versioning

Concept packs use semantic versioning:
- **Major**: Breaking changes to ontology structure
- **Minor**: New domain extensions, more models
- **Patch**: Bug fixes, lens retraining

## Community Workflow

1. **User trains lenses** for their model on a concept pack
2. **Validates** lens quality (F1 scores, etc.)
3. **Uploads** to HuggingFace: `hatcat/{pack_id}`
4. **Others download** and use via pack_id reference
5. **Contribute back** improved lenses or new domain extensions

## Migration Path

Current structure → Concept pack:
1. Create `pack.json` for existing SUMO setup
2. Move lenses to `concept_packs/sumo-wordnet-aisafety-v1/models/gemma-3-4b-pt/`
3. Update server to read from concept pack structure
4. Add API endpoints for pack discovery
