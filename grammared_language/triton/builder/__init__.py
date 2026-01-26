"""Triton Repository Builder Package."""

import importlib
import pkgutil
from pathlib import Path

# Programmatically discover and import all modules
__all__ = []
_module_path = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(_module_path)]):
    if not module_name.startswith('_'):
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # Export all public attributes from the module
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                # Only export classes and functions defined in this module
                if hasattr(attr, '__module__') and attr.__module__.startswith(__name__):
                    globals()[attr_name] = attr
                    __all__.append(attr_name)
