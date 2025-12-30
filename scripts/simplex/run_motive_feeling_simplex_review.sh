#!/bin/bash
# Run simplex agentic review for 470 motive/feeling concepts

set -e

# Temporarily symlink the motive/feeling file to where the script expects it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"

MOTIVE_FEELING_FILE="$RESULTS_DIR/motive_feeling_tier2_format.json"
TIER2_LINK="$RESULTS_DIR/tier2_scoring_revised/tier2_top500_concepts_revised.json"
BACKUP_FILE="${TIER2_LINK}.backup_before_motive_feeling"
OUTPUT_FILE="$RESULTS_DIR/simplex_agentic_review.json"
MOTIVE_OUTPUT="$RESULTS_DIR/motive_feeling_simplex_review.json"

echo "=================================================="
echo "Motive/Feeling Simplex Review (470 concepts)"
echo "=================================================="

# Backup original tier2 file if it exists
if [ -f "$TIER2_LINK" ]; then
    echo "Backing up original tier2 file..."
    cp "$TIER2_LINK" "$BACKUP_FILE"
fi

# Backup original output file if it exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Backing up original simplex review results..."
    cp "$OUTPUT_FILE" "${OUTPUT_FILE}.backup_before_motive_feeling"
fi

# Create symlink to motive/feeling file
echo "Linking motive/feeling concepts to tier2 location..."
mkdir -p "$(dirname "$TIER2_LINK")"
cp "$MOTIVE_FEELING_FILE" "$TIER2_LINK"

# Run the review
echo ""
echo "Starting simplex review for 470 motive/feeling concepts..."
echo "Estimated cost: ~\$14.10 (470 * \$0.03)"
echo "Estimated time: ~23.5 minutes"
echo ""

cd "$PROJECT_ROOT"
. .venv/bin/activate

# Ensure API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable not set"
    exit 1
fi

export ANTHROPIC_API_KEY
python scripts/run_simplex_agentic_review.py 470 <<< 'y' 2>&1 | tee results/motive_feeling_simplex_review.log

# Save output to motive-specific file
if [ -f "$OUTPUT_FILE" ]; then
    mv "$OUTPUT_FILE" "$MOTIVE_OUTPUT"
    echo ""
    echo "Results saved to: $MOTIVE_OUTPUT"
fi

# Restore original tier2 file
if [ -f "$BACKUP_FILE" ]; then
    echo "Restoring original tier2 file..."
    mv "$BACKUP_FILE" "$TIER2_LINK"
fi

# Restore original output file
if [ -f "${OUTPUT_FILE}.backup_before_motive_feeling" ]; then
    echo "Restoring original simplex review results..."
    mv "${OUTPUT_FILE}.backup_before_motive_feeling" "$OUTPUT_FILE"
fi

echo ""
echo "✅ Motive/Feeling simplex review complete!"
echo "Results: $MOTIVE_OUTPUT"
echo "Log: results/motive_feeling_simplex_review.log"
