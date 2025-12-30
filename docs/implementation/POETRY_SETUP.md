# Poetry Setup Guide for HatCat

This project uses Poetry for dependency management in externally managed Python environments.

## Quick Setup

```bash
# Run the setup script (installs Poetry if needed and sets up dependencies)
./setup.sh
```

## Manual Setup

### 1. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Install Dependencies

```bash
# Install all dependencies (creates .venv in project directory)
poetry install --no-root

# Or if you want to install HatCat as a package:
poetry install
```

### 3. Activate Environment

```bash
# Option 1: Spawn a shell within the virtual environment
poetry shell

# Option 2: Run commands with poetry run
poetry run python scripts/validate_setup.py
```

## Usage

### Running Scripts

```bash
# With poetry shell activated
python scripts/convergence_validation.py --concepts democracy dog running

# Or directly with poetry run
poetry run python scripts/convergence_validation.py --concepts democracy dog running
```

### Running Week 2 Pipeline

```bash
# Day 1: Convergence validation
poetry run python scripts/convergence_validation.py \
    --concepts democracy dog running happiness gravity justice freedom computer learning mountain

# Day 2-3: Bootstrap and train
poetry run python scripts/stage_0_bootstrap.py --n-concepts 1000 --output data/processed/encyclopedia_stage0_1k.h5
poetry run python scripts/train_interpreter.py --data data/processed/encyclopedia_stage0_1k.h5 --epochs 10

# Day 5: Scale to 50K
poetry run python scripts/stage_0_bootstrap.py --n-concepts 50000 --output data/processed/encyclopedia_stage0_full.h5 --layers -12 -9 -6 -3 -1
```

### Running Tests

```bash
poetry run python tests/test_activation_capture.py
poetry run python scripts/capture_concepts.py
```

## Poetry Commands

### Managing Dependencies

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

### Environment Management

```bash
# Show virtual environment info
poetry env info

# List all virtual environments
poetry env list

# Remove virtual environment
poetry env remove python3.10

# Use specific Python version
poetry env use python3.10
```

### Lock File

```bash
# Update poetry.lock without installing
poetry lock

# Install from lock file (reproducible builds)
poetry install --no-root
```

## Externally Managed Environment

This project is configured to work with externally managed Python environments (common on Linux systems like Ubuntu 24.04+).

### Configuration

The setup script automatically configures:
```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

This creates a `.venv` directory in the project root, isolating dependencies from the system Python.

### Why `--no-root`?

The `--no-root` flag prevents Poetry from trying to install the project itself as a package. This is useful when:
- You're working on a project that's not a library
- You want to avoid installation conflicts in externally managed environments
- You just want to install dependencies without package installation

If you want HatCat importable as a package, omit `--no-root`:
```bash
poetry install
```

## Integration with IDEs

### VSCode

Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

## Troubleshooting

### "externally-managed-environment" Error

If you get this error, it means your system Python is managed by the OS package manager. This is why we use Poetry with `virtualenvs.create true`.

**Solution**: Make sure Poetry is configured to create virtual environments:
```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
poetry install --no-root
```

### Poetry Not Found After Installation

**Solution**: Add Poetry to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Make it permanent by adding to `~/.bashrc`:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### "poetry: command not found" in Scripts

**Solution**: Use `poetry run` to execute scripts:
```bash
poetry run python scripts/validate_setup.py
```

Or activate the shell first:
```bash
poetry shell
python scripts/validate_setup.py
```

### Dependencies Taking Too Long to Install

PyTorch is large (~2GB). First installation will take time. Subsequent installs use cached wheels.

**Speed up**: Install with `--no-dev` to skip development dependencies:
```bash
poetry install --no-root --no-dev
```

### Virtual Environment in Wrong Location

**Solution**: Configure Poetry to create .venv in project directory:
```bash
poetry config virtualenvs.in-project true
poetry install --no-root
```

### Lock File Out of Sync

If `poetry.lock` is out of sync with `pyproject.toml`:
```bash
poetry lock --no-update
poetry install --no-root
```

## Comparison: pip vs Poetry

| Feature | pip + venv | Poetry |
|---------|------------|--------|
| Dependency resolution | Manual | Automatic |
| Lock file | requirements.txt | poetry.lock |
| Dev dependencies | Separate file | Built-in groups |
| Virtual env creation | Manual | Automatic |
| Reproducible builds | Approximate | Exact |
| Dependency tree | No | Yes |
| Project metadata | setup.py | pyproject.toml |

## Migration from requirements.txt

The `requirements.txt` is kept for backward compatibility, but Poetry uses `pyproject.toml`.

To update requirements.txt from Poetry:
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Advanced Usage

### Multiple Python Versions

```bash
# Use specific Python version
poetry env use 3.10
poetry env use 3.11

# List available environments
poetry env list

# Remove all environments
poetry env remove --all
```

### Custom Configuration

```bash
# Set cache directory
poetry config cache-dir /path/to/cache

# Disable virtualenv creation (use system Python)
poetry config virtualenvs.create false  # NOT recommended

# Set virtualenv location
poetry config virtualenvs.path /path/to/venvs
```

### CI/CD Integration

```bash
# GitHub Actions
poetry install --no-root --no-interaction --no-ansi

# Docker
RUN pip install poetry && poetry install --no-root --no-dev
```

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry Basic Usage](https://python-poetry.org/docs/basic-usage/)
- [Dependency Management](https://python-poetry.org/docs/managing-dependencies/)
- [Configuration](https://python-poetry.org/docs/configuration/)

## Quick Reference

```bash
# Setup
./setup.sh                          # Run setup script
poetry install --no-root            # Install dependencies

# Activate
poetry shell                        # Spawn shell in venv
poetry run <command>                # Run command in venv

# Week 2 Workflow
poetry run python scripts/convergence_validation.py ...
poetry run python scripts/stage_0_bootstrap.py ...
poetry run python scripts/train_interpreter.py ...

# Dependency management
poetry add <package>                # Add dependency
poetry remove <package>             # Remove dependency
poetry update                       # Update dependencies

# Environment
poetry env info                     # Show venv info
poetry env list                     # List venvs
poetry env remove <python>          # Remove venv
```
