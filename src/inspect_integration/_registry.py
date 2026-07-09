"""
Task registry for Inspect discovery.

This module registers HatCat tasks with Inspect's task discovery system.
"""

from typing import Any, Dict, List, Optional

# Inspect imports
try:
    from inspect_ai import Task
    from inspect_ai._eval.registry import task_register
    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False
    Task = object
    task_register = lambda *args, **kwargs: lambda f: f


def register_hatcat_tasks():
    """
    Register HatCat tasks with Inspect's registry.

    This enables discovery via inspect eval commands.
    """
    if not INSPECT_AVAILABLE:
        return

    from .tasks import hatcat_wrapped

    # The @task decorator already handles registration
    # This function is called at module import to ensure tasks are registered


def get_registered_tasks() -> Dict[str, Any]:
    """
    Get all registered HatCat tasks.

    Returns:
        Dict mapping task names to task factories
    """
    return {
        "hatcat_wrapped": "src.inspect_integration.tasks.wrapped:hatcat_wrapped",
    }


# Auto-register on import
if INSPECT_AVAILABLE:
    register_hatcat_tasks()
