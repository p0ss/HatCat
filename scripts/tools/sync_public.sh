#!/bin/bash
# Sync between HatCatDev (private) and HatCat (public)
#
# IMPORTANT: This copies FILES, not git history, to avoid leaking private data.
#
# Usage:
#   ./sync_public.sh pull    # Pull changes from public HatCat into HatCatDev (for PRs)
#   ./sync_public.sh push    # Copy shared files from HatCatDev to HatCat and push
#   ./sync_public.sh status  # Show what's different between repos
#   ./sync_public.sh diff    # Show detailed file differences

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

# concept_packs subfolder that's shared (just this one, not all concept_packs)
SHARED_CONCEPT_PACK="concept_packs/first-light"

# Common rsync excludes
RSYNC_EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.pytest_cache'
    --exclude='*.egg-info'
)

sync_dev_to_public() {
    echo "Syncing files from HatCatDev to HatCat..."

    # Sync shared directories
    for dir in "${SHARED_DIRS[@]}"; do
        if [ -d "$HATCATDEV_DIR/$dir" ]; then
            echo "  $dir/"
            rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
                "$HATCATDEV_DIR/$dir/" "$HATCAT_DIR/$dir/"
        fi
    done

    # Sync shared root files
    for file in "${SHARED_FILES[@]}"; do
        if [ -f "$HATCATDEV_DIR/$file" ]; then
            echo "  $file"
            cp "$HATCATDEV_DIR/$file" "$HATCAT_DIR/$file"
        fi
    done

    # Sync the one shared concept pack
    if [ -d "$HATCATDEV_DIR/$SHARED_CONCEPT_PACK" ]; then
        echo "  $SHARED_CONCEPT_PACK/"
        mkdir -p "$HATCAT_DIR/concept_packs"
        rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
            "$HATCATDEV_DIR/$SHARED_CONCEPT_PACK/" "$HATCAT_DIR/$SHARED_CONCEPT_PACK/"
    fi
}

sync_public_to_dev() {
    echo "Syncing files from HatCat to HatCatDev..."

    # Sync shared directories
    for dir in "${SHARED_DIRS[@]}"; do
        if [ -d "$HATCAT_DIR/$dir" ]; then
            echo "  $dir/"
            rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
                "$HATCAT_DIR/$dir/" "$HATCATDEV_DIR/$dir/"
        fi
    done

    # Sync shared root files
    for file in "${SHARED_FILES[@]}"; do
        if [ -f "$HATCAT_DIR/$file" ]; then
            echo "  $file"
            cp "$HATCAT_DIR/$file" "$HATCATDEV_DIR/$file"
        fi
    done

    # Sync the one shared concept pack
    if [ -d "$HATCAT_DIR/$SHARED_CONCEPT_PACK" ]; then
        echo "  $SHARED_CONCEPT_PACK/"
        rsync -av --delete "${RSYNC_EXCLUDES[@]}" \
            "$HATCAT_DIR/$SHARED_CONCEPT_PACK/" "$HATCATDEV_DIR/$SHARED_CONCEPT_PACK/"
    fi
}

show_diff_count() {
    local total=0

    for dir in "${SHARED_DIRS[@]}"; do
        if [ -d "$HATCATDEV_DIR/$dir" ] && [ -d "$HATCAT_DIR/$dir" ]; then
            count=$(diff -rq "$HATCATDEV_DIR/$dir" "$HATCAT_DIR/$dir" 2>/dev/null | \
                grep -v __pycache__ | grep -v ".pyc" | wc -l)
            if [ "$count" -gt 0 ]; then
                echo "  $dir: $count files differ"
                total=$((total + count))
            fi
        fi
    done

    for file in "${SHARED_FILES[@]}"; do
        if [ -f "$HATCATDEV_DIR/$file" ] && [ -f "$HATCAT_DIR/$file" ]; then
            if ! diff -q "$HATCATDEV_DIR/$file" "$HATCAT_DIR/$file" >/dev/null 2>&1; then
                echo "  $file: differs"
                total=$((total + 1))
            fi
        fi
    done

    if [ -d "$HATCATDEV_DIR/$SHARED_CONCEPT_PACK" ] && [ -d "$HATCAT_DIR/$SHARED_CONCEPT_PACK" ]; then
        count=$(diff -rq "$HATCATDEV_DIR/$SHARED_CONCEPT_PACK" "$HATCAT_DIR/$SHARED_CONCEPT_PACK" 2>/dev/null | \
            grep -v __pycache__ | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "  $SHARED_CONCEPT_PACK: $count files differ"
            total=$((total + count))
        fi
    fi

    echo ""
    echo "Total: $total file(s) differ"
    return $total
}

case "$1" in
    pull)
        echo "=== Pulling from public (HatCat) into HatCatDev ==="
        echo "This copies changed files from HatCat into HatCatDev"
        echo ""

        # Check for uncommitted changes in HatCatDev
        cd "$HATCATDEV_DIR"
        if ! git diff-index --quiet HEAD -- 2>/dev/null; then
            echo "WARNING: You have uncommitted changes in HatCatDev."
            echo "Consider committing or stashing them first."
            echo ""
        fi

        echo "Current differences:"
        show_diff_count || true
        echo ""

        read -p "Proceed with sync? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sync_public_to_dev
            echo ""
            echo "Done! Review changes in HatCatDev with 'git status' and commit when ready."
        else
            echo "Aborted."
        fi
        ;;

    push)
        echo "=== Pushing from HatCatDev to public (HatCat) ==="
        echo "This copies shared files to HatCat and commits them"
        echo ""

        # Check for uncommitted changes in HatCatDev
        cd "$HATCATDEV_DIR"
        if ! git diff-index --quiet HEAD -- 2>/dev/null; then
            echo "ERROR: You have uncommitted changes in HatCatDev."
            echo "Commit them first so the sync is based on a known state."
            exit 1
        fi

        echo "Current differences:"
        show_diff_count || true
        echo ""

        echo "This will:"
        echo "  1. Copy shared files from HatCatDev to HatCat"
        echo "  2. Commit changes in HatCat"
        echo "  3. Push to origin"
        echo ""
        echo "ONLY these paths will be synced:"
        for dir in "${SHARED_DIRS[@]}"; do echo "  - $dir/"; done
        for file in "${SHARED_FILES[@]}"; do echo "  - $file"; done
        echo "  - $SHARED_CONCEPT_PACK/"
        echo ""

        read -p "Proceed? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sync_dev_to_public

            echo ""
            echo "Files synced. Now committing in HatCat..."

            cd "$HATCAT_DIR"

            # Check if there are changes to commit
            if git diff-index --quiet HEAD -- 2>/dev/null; then
                echo "No changes to commit - repos are already in sync."
                exit 0
            fi

            git add -A

            # Get the latest commit message from HatCatDev for reference
            LATEST_MSG=$(cd "$HATCATDEV_DIR" && git log -1 --format="%s")

            echo ""
            echo "Changes to commit:"
            git status --short
            echo ""

            read -p "Commit message (or Enter for 'Sync: $LATEST_MSG'): " COMMIT_MSG
            if [ -z "$COMMIT_MSG" ]; then
                COMMIT_MSG="Sync: $LATEST_MSG"
            fi

            git commit -m "$COMMIT_MSG"

            echo ""
            read -p "Push to origin? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git push origin main
                echo "Done! Changes pushed to public HatCat repo."
            else
                echo "Committed locally but not pushed. Run 'git push origin main' in HatCat when ready."
            fi
        else
            echo "Aborted."
        fi
        ;;

    status)
        echo "=== Sync Status ==="
        echo ""
        echo "HatCatDev: $HATCATDEV_DIR"
        echo "HatCat:    $HATCAT_DIR"
        echo ""
        echo "Shared paths:"
        for dir in "${SHARED_DIRS[@]}"; do echo "  - $dir/"; done
        for file in "${SHARED_FILES[@]}"; do echo "  - $file"; done
        echo "  - $SHARED_CONCEPT_PACK/"
        echo ""
        echo "File differences:"
        show_diff_count || true
        ;;

    diff)
        echo "=== Detailed Diff ==="
        echo ""
        for dir in "${SHARED_DIRS[@]}"; do
            if [ -d "$HATCATDEV_DIR/$dir" ] && [ -d "$HATCAT_DIR/$dir" ]; then
                diff -rq "$HATCATDEV_DIR/$dir" "$HATCAT_DIR/$dir" 2>/dev/null | \
                    grep -v __pycache__ | grep -v ".pyc" || true
            fi
        done
        ;;

    *)
        echo "Usage: $0 {pull|push|status|diff}"
        echo ""
        echo "  push   - Copy shared files from HatCatDev to HatCat and push"
        echo "  pull   - Copy shared files from HatCat to HatCatDev (for PRs)"
        echo "  status - Show sync status between repos"
        echo "  diff   - Show detailed file differences"
        echo ""
        echo "IMPORTANT: This syncs FILES only, not git history."
        echo "Private directories (lens_packs, data, results) are never synced."
        exit 1
        ;;
esac
