"""
System Test Module.

Tests the dynamic loading of modules.
"""

import importlib.util
from typing import Any


def load_module(module_name: str) -> Any:
    """
    Dynamically loads a module by its name.

    Args:
        module_name: Name of the module to load.

    Returns:
        The loaded module.

    Raises:
        ImportError: If the module cannot be located.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Module {module_name} not found")
    module = importlib.util.module_from_spec(spec)
    # type: ignore  # spec.loader is assumed non-None because of the check above.
    spec.loader.exec_module(module)
    return module


def test_load_module() -> None:
    mod = load_module("os")
    assert hasattr(mod, "path")


if __name__ == "__main__":
    test_load_module()
    print("Module load test passed.")
