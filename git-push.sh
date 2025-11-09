#!/bin/bash
# Helper script to push to GitHub using HATCAT_GITHUB_TOKEN
# Usage: ./git-push.sh

set -e

if [ -z "$HATCAT_GITHUB_TOKEN" ]; then
    echo "Error: HATCAT_GITHUB_TOKEN environment variable not set"
    echo ""
    echo "Please set it first:"
    echo "  export HATCAT_GITHUB_TOKEN='your_token_here'"
    echo ""
    echo "Then run this script again:"
    echo "  ./git-push.sh"
    exit 1
fi

echo "Configuring git remote with token..."
git remote set-url origin "https://${HATCAT_GITHUB_TOKEN}@github.com/p0ss/HatCat.git"

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✓ Successfully pushed to GitHub!"
echo ""

# Optionally remove token from remote URL for security
read -p "Remove token from git config for security? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git remote set-url origin "https://github.com/p0ss/HatCat.git"
    echo "✓ Token removed from git config"
fi
