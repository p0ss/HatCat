# SUMO Source Files

This directory contains the authoritative SUMO (Suggested Upper Merged Ontology) source files used to build the hierarchical concept layers.

## Files

### Merge.kif (621 KB)
- **Source**: https://raw.githubusercontent.com/ontologyportal/sumo/master/Merge.kif
- **Format**: SUO-KIF (Standard Upper Ontology - Knowledge Interchange Format)
- **Content**: Core SUMO ontology with 684 classes and 805 subclass relations
- **Hierarchy**: Entity → Physical/Abstract → ... (max depth 10)
- **License**: IEEE Standard Upper Ontology

### SUMO.owl (36 MB)
- **Source**: https://www.ontologyportal.com/SUMO.owl.html
- **Format**: OWL/RDF (Web Ontology Language)
- **Status**: ⚠️ "Provisional and necessarily lossy translation" - NOT USED
- **Issue**: Incomplete hierarchy (3,202 orphan roots, Entity isolated at depth 0)
- **Note**: Kept for reference only - Merge.kif is the authoritative source

## Usage

The `src/build_sumo_wordnet_layers_v5.py` script parses `Merge.kif` to extract the SUMO class hierarchy, merges in the custom AI/safety domains, and joins everything with WordNet mappings to create the 7 abstraction layers used across the project.

## License

SUMO is released under the IEEE Standard Upper Ontology license. See:
- http://www.ontologyportal.org/
- Original paper: Niles & Pease (2001) "Towards a Standard Upper Ontology"

## Citation

If using SUMO in research, cite:

```
Niles, I., and Pease, A. 2001. Towards a Standard Upper Ontology.
In Proceedings of the 2nd International Conference on Formal Ontology
in Information Systems (FOIS-2001), Ogunquit, Maine, October 17-19, 2001.
```
