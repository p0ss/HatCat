# Poetry Migration Complete ✅

The HatCat project now uses **Poetry** for dependency management, optimized for externally managed Python environments.

## What Changed

### Files Added
- ✅ `pyproject.toml` - Poetry configuration and dependencies
- ✅ `POETRY_SETUP.md` - Comprehensive Poetry usage guide

### Files Modified
- ✅ `setup.sh` - Now installs and configures Poetry
- ✅ `README.md` - Updated installation instructions
- ✅ `QUICKSTART.md` - Updated with Poetry commands

### Files Kept (Backward Compatibility)
- ✅ `requirements.txt` - Still available for pip users

## Quick Start

```bash
# Setup (auto-installs Poetry if needed)
./setup.sh

# Activate environment
poetry shell

# Run Week 2 pipeline
poetry run python scripts/convergence_validation.py --concepts democracy dog running
poetry run python scripts/stage_0_bootstrap.py --n-concepts 1000
poetry run python scripts/train_interpreter.py --data data/processed/encyclopedia_stage0_1k.h5
```

## Why Poetry?

### For Externally Managed Environments

Modern Linux distributions (Ubuntu 24.04+, Fedora 40+) mark system Python as "externally managed" to prevent conflicts with OS package managers. Poetry handles this elegantly by:

1. **Automatic Virtual Environment**: Creates isolated `.venv` in project directory
2. **Dependency Resolution**: Solves complex dependency chains automatically
3. **Lock File**: `poetry.lock` ensures reproducible builds
4. **Development Groups**: Separates dev dependencies from production

### Benefits Over pip + venv

| Feature | pip + venv | Poetry |
|---------|-----------|--------|
| Dependency resolution | Manual | ✅ Automatic |
| Lock file | requirements.txt | ✅ poetry.lock (exact versions) |
| Dev dependencies | Separate file | ✅ Built-in groups |
| Virtual env creation | Manual | ✅ Automatic |
| Reproducible builds | Approximate | ✅ Exact |
| Dependency tree | ❌ No | ✅ Yes (`poetry show --tree`) |

## Commands Comparison

### Setup
```bash
# Old (pip)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# New (Poetry)
./setup.sh
poetry shell
```

### Running Scripts
```bash
# Old (pip)
source venv/bin/activate
python scripts/convergence_validation.py --concepts democracy dog

# New (Poetry)
poetry run python scripts/convergence_validation.py --concepts democracy dog

# Or after poetry shell
python scripts/convergence_validation.py --concepts democracy dog
```

### Adding Dependencies
```bash
# Old (pip)
pip install new-package
pip freeze > requirements.txt  # Manual update

# New (Poetry)
poetry add new-package  # Auto-updates pyproject.toml and poetry.lock
```

## Migration Notes

### For Existing Users

If you already have a pip-based setup:

```bash
# Option 1: Start fresh with Poetry
rm -rf venv
./setup.sh

# Option 2: Keep pip setup
# Everything still works! requirements.txt is maintained.
```

### For New Users

Just run:
```bash
./setup.sh
```

That's it! Poetry will be installed automatically if not present.

## Configuration

Poetry is configured for in-project virtual environments:

```toml
# pyproject.toml
[tool.poetry]
name = "hatcat"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.8"
torch = "^2.0.0"
transformers = "^4.30.0"
# ... etc
```

The setup script configures:
```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

This creates `.venv/` in the project root for isolation.

## Troubleshooting

### "Poetry not found after installation"

Add to your `~/.bashrc`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then:
```bash
source ~/.bashrc
```

### "externally-managed-environment" error

This is expected on modern Linux. Poetry handles it automatically with its virtual environment.

**Don't use**:
```bash
pip install --break-system-packages  # DON'T DO THIS
```

**Instead use**:
```bash
poetry install --no-root
```

### Slow dependency installation

PyTorch is ~2GB. First install takes time. Poetry caches wheels for future installs.

Speed up development installs:
```bash
poetry install --no-root --no-dev  # Skip dev dependencies
```

## Documentation

- **POETRY_SETUP.md** - Complete Poetry guide with troubleshooting
- **README.md** - Updated installation section
- **QUICKSTART.md** - Week 2 workflow with Poetry commands

## Backward Compatibility

The `requirements.txt` is kept for users who prefer pip. To update it from Poetry:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Week 2 Workflow (Poetry Version)

```bash
# Setup
./setup.sh
poetry shell

# Day 1: Convergence validation (30 min)
poetry run python scripts/convergence_validation.py \
    --concepts democracy dog running happiness gravity justice freedom

# Day 2-3: Bootstrap and train (20 min)
poetry run python scripts/stage_0_bootstrap.py --n-concepts 1000
poetry run python scripts/train_interpreter.py --data data/processed/encyclopedia_stage0_1k.h5 --epochs 10

# Day 5: Scale to 50K (45 min)
poetry run python scripts/stage_0_bootstrap.py --n-concepts 50000 --layers -12 -9 -6 -3 -1
```

## Summary

✅ **Poetry is now the recommended setup method**
✅ **pip + venv still works** (backward compatible)
✅ **Automatic Poetry installation** via `./setup.sh`
✅ **Optimized for externally managed environments**
✅ **All documentation updated**

---

**Migration Date**: November 1, 2024
**Poetry Version**: Latest (auto-installed)
**Python Support**: 3.8+
