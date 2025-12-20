"""
UI - User Interface Layer

Application layer providing various interfaces to the FTW system.

Submodules:
- ui.openwebui: OpenWebUI pipeline integration
- ui.streamlit: Streamlit-based visualization apps
- ui.visualization: Visualization utilities and color mapping
"""

# Re-export visualization utilities
from .visualization import ConceptColorMapper, get_color_mapper

__all__ = [
    'ConceptColorMapper',
    'get_color_mapper',
]
