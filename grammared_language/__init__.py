"""Grammared Language - Grammar correction tools and models."""

import importlib
import pkgutil
from pathlib import Path

__version__ = "0.1.0"

# Programmatically discover and import all submodules
__all__ = ["__version__"]
_module_path = Path(__file__).parent

# Import subpackages
for _, module_name, is_pkg in pkgutil.iter_modules([str(_module_path)]):
    if not module_name.startswith('_') and is_pkg:
        try:
            module = importlib.import_module(f'.{module_name}', package=__name__)
            globals()[module_name] = module
            __all__.append(module_name)
        except ImportError as e:
            # Skip modules that have missing dependencies
            pass
