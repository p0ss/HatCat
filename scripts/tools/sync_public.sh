#!/bin/bash
# Sync between HatCatDev (private) and HatCat (public)
#
# Usage:
#   ./sync_public.sh pull    # Pull changes from public HatCat into HatCatDev (for PRs)
#   ./sync_public.sh push    # Push changes from HatCatDev to public HatCat
#   ./sync_public.sh status  # Show what's different between repos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HATCATDEV_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
HATCAT_DIR="/home/poss/Documents/Code/HatCat"

# Directories that exist in both repos (the shared subset)
SHARED_DIRS=(
    "src"
    "scripts"
    "docs"
    "melds"
    "tests"
    "img"
)

# Files at root that are shared
SHARED_FILES=(
    "pyproject.toml"
    "poetry.lock"
    "requirements.txt"
    "README.md"
    "setup.sh"
    "start_hatcat_ui.sh"
)

# concept_packs subfolder that's shared
SHARED_CONCEPT_PACK="concept_packs/first-light"

cd "$HATCATDEV_DIR"

case "$1" in
    pull)
        echo "=== Pulling from public (HatCat) into HatCatDev ==="
        echo "This merges any PR changes from HatCat into your local HatCatDev"
        echo ""

        git fetch public

        # Show what would be merged
        echo "Commits in public/main not in current branch:"
        git log --oneline HEAD..public/main 2>/dev/null || echo "  (none or branches diverged)"
        echo ""

        read -p "Proceed with merge? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git merge public/main -m "Merge public HatCat contributions"
            echo "Done! Review changes and push to origin when ready."
        else
            echo "Aborted."
        fi
        ;;

    push)
        echo "=== Pushing from HatCatDev to public (HatCat) ==="
        echo "This syncs your changes to the public repo"
        echo ""

        # Check for uncommitted changes
        if ! git diff-index --quiet HEAD --; then
            echo "ERROR: You have uncommitted changes. Commit or stash them first."
            exit 1
        fi

        git fetch public

        # Show what would be pushed
        echo "Commits in HatCatDev not in public/main:"
        git log --oneline public/main..HEAD -- ${SHARED_DIRS[@]} ${SHARED_FILES[@]} "$SHARED_CONCEPT_PACK" 2>/dev/null | head -20
        echo ""

        # Check if public is behind
        BEHIND=$(git rev-list --count public/main..HEAD -- ${SHARED_DIRS[@]} ${SHARED_FILES[@]} "$SHARED_CONCEPT_PACK" 2>/dev/null || echo "0")

        if [ "$BEHIND" = "0" ]; then
            echo "Public repo is up to date with shared directories."
            exit 0
        fi

        echo "Will push $BEHIND commits affecting shared directories."
        echo ""
        echo "NOTE: This will push to public/main. Make sure you've tested!"
        read -p "Proceed? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Push current branch to public's main
            # Using subtree split would be cleaner but this works for subset
            git push public HEAD:main
            echo "Done! Changes pushed to public HatCat repo."
        else
            echo "Aborted."
        fi
        ;;

    status)
        echo "=== Sync Status ==="
        echo ""

        git fetch public 2>/dev/null

        echo "HatCatDev branch: $(git branch --show-current)"
        echo "HatCatDev HEAD:   $(git rev-parse --short HEAD)"
        echo ""

        echo "Public (HatCat) main: $(git rev-parse --short public/main 2>/dev/null || echo 'unknown')"
        echo ""

        echo "Commits in HatCatDev not in public:"
        git log --oneline public/main..HEAD -- ${SHARED_DIRS[@]} ${SHARED_FILES[@]} "$SHARED_CONCEPT_PACK" 2>/dev/null | head -10 || echo "  (none)"
        echo ""

        echo "Commits in public not in HatCatDev:"
        git log --oneline HEAD..public/main 2>/dev/null | head -10 || echo "  (none)"
        echo ""

        echo "File differences in shared directories:"
        for dir in "${SHARED_DIRS[@]}"; do
            DIFF_COUNT=$(diff -rq "$HATCATDEV_DIR/$dir" "$HATCAT_DIR/$dir" 2>/dev/null | grep -v __pycache__ | grep -v ".pyc" | wc -l)
            if [ "$DIFF_COUNT" -gt 0 ]; then
                echo "  $dir: $DIFF_COUNT files differ"
            fi
        done
        ;;

    *)
        echo "Usage: $0 {pull|push|status}"
        echo ""
        echo "  pull   - Pull PR contributions from public HatCat into HatCatDev"
        echo "  push   - Push your HatCatDev changes to public HatCat"
        echo "  status - Show sync status between repos"
        exit 1
        ;;
esac
