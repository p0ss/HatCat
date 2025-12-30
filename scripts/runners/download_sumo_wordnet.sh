#!/bin/bash
# Download SUMO ontology files and WordNet 3.0 mappings

set -e

SUMO_DIR="data/concept_graph/sumo_source"
mkdir -p "$SUMO_DIR"

echo "Downloading SUMO ontology files..."

# SUMO base URL
SUMO_BASE="https://raw.githubusercontent.com/ontologyportal/sumo/master"

# Download core ontology files
wget -q -O "$SUMO_DIR/Merge.kif" "$SUMO_BASE/Merge.kif" || echo "Merge.kif already exists or download failed"
wget -q -O "$SUMO_DIR/Mid-level-ontology.kif" "$SUMO_BASE/Mid-level-ontology.kif" || echo "Downloading Mid-level-ontology.kif..."

# Download domain ontologies (including emotion)
wget -q -O "$SUMO_DIR/emotion.kif" "$SUMO_BASE/emotion.kif" || echo "Downloading emotion.kif..."

# Download WordNet 3.0 mappings
echo "Downloading WordNet 3.0 mappings..."
MAPPING_BASE="$SUMO_BASE/WordNetMappings"
wget -q -O "$SUMO_DIR/WordNetMappings30-noun.txt" "$MAPPING_BASE/WordNetMappings30-noun.txt" || echo "Failed to download WordNetMappings30-noun.txt"
wget -q -O "$SUMO_DIR/WordNetMappings30-verb.txt" "$MAPPING_BASE/WordNetMappings30-verb.txt" || echo "Failed to download WordNetMappings30-verb.txt"
wget -q -O "$SUMO_DIR/WordNetMappings30-adj.txt" "$MAPPING_BASE/WordNetMappings30-adj.txt" || echo "Failed to download WordNetMappings30-adj.txt"
wget -q -O "$SUMO_DIR/WordNetMappings30-adv.txt" "$MAPPING_BASE/WordNetMappings30-adv.txt" || echo "Failed to download WordNetMappings30-adv.txt"

echo "✓ Download complete"
ls -lh "$SUMO_DIR"
