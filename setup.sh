#!/bin/bash
# Setup script for HatCat project using Poetry
# For externally managed Python environments

set -e  # Exit on error

echo "=================================="
echo "HatCat Setup (Poetry)"
echo "=================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo ""
    echo "Poetry not found. Installing Poetry..."
    echo ""

    # Install Poetry using the official installer
    curl -sSL https://install.python-poetry.org | python3 -

    # Add Poetry to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    echo ""
    echo "✓ Poetry installed successfully"
    echo ""
    echo "NOTE: You may need to add Poetry to your PATH permanently:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "  (Add this to your ~/.bashrc or ~/.zshrc)"
    echo ""
else
    echo "✓ Poetry is already installed"
    poetry --version
fi

# Configure Poetry for externally managed environments
echo ""
echo "Configuring Poetry..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes (downloading PyTorch, etc.)..."
echo ""

# Install dependencies without creating a venv (uses system Python)
# For externally managed environments, we use --no-root to avoid installation issues
poetry install --no-root

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To activate the Poetry environment:"
echo "  poetry shell"
echo ""
echo "Or run commands directly:"
echo "  poetry run python scripts/validate_setup.py"
echo ""
echo "Next steps:"
echo ""
echo "1. Launch Streamlit UI:"
echo "   poetry run streamlit run src/ui/streamlit_chat.py"
echo ""
echo "2. Validate setup:"
echo "   poetry run python scripts/validate_setup.py"
echo ""
echo "3. Run tests:"
echo "   poetry run python tests/test_activation_capture.py"
echo ""
echo "4. See QUICKSTART.md for advanced workflows"
echo ""
