"""Triton Inference Server model implementations."""

import importlib
import pkgutil
from pathlib import Path

try:
    import triton_python_backend_utils as pb_utils
    is_triton_server = True
except ImportError:
    is_triton_server = False


# Programmatically discover and import all modules
__all__ = []
_module_path = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(_module_path)]):
    if not module_name.startswith('_'):
        if module_name.startswith('triton_') and not is_triton_server:
            # skips Triton-specific models when not in Triton server environment
            continue
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # Export all public attributes from the module
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                # Only export classes and functions defined in this module
                if hasattr(attr, '__module__') and attr.__module__.startswith(__name__):
                    globals()[attr_name] = attr
                    __all__.append(attr_name)
