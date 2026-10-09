"""Compatibility shims for third-party matcher code.

kornia >= 0.8.3 moved several helpers out of ``kornia.utils`` (into
``kornia.geometry`` and ``kornia.image``), leaving deprecated wrappers that
warn on every call, and dropped the public ``kornia.utils.grid`` submodule.
Third-party code (e.g. ASpanFormer, kornia_moons) still uses the old names, so
they are pointed to the new functions when these exist, and
``kornia.utils.grid`` is recreated. Imported by ``core`` before any matcher.
"""

import importlib
import sys
import types

import kornia
import kornia.utils

for _name, _new in [
    ("create_meshgrid", "kornia.geometry"),
    ("create_meshgrid3d", "kornia.geometry"),
    ("image_to_tensor", "kornia.image"),
    ("tensor_to_image", "kornia.image"),
]:
    try:
        _fn = getattr(importlib.import_module(_new), _name)
    except (ImportError, AttributeError):
        continue
    for _mod_name, _mod in list(sys.modules.items()):
        if (_mod_name == "kornia" or _mod_name.startswith("kornia.utils")) and hasattr(_mod, _name):
            setattr(_mod, _name, _fn)

if "kornia.utils.grid" not in sys.modules:
    _grid = types.ModuleType("kornia.utils.grid")
    for _name in ("create_meshgrid", "create_meshgrid3d"):
        _fn = getattr(kornia.utils, _name, None)
        if _fn is not None:
            setattr(_grid, _name, _fn)
    sys.modules["kornia.utils.grid"] = _grid
    kornia.utils.grid = _grid
