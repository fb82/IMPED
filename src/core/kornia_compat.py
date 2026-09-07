"""Compatibility shims for third-party matcher code.

kornia >= 0.8.3 dropped the public ``kornia.utils.grid`` submodule, but several
"""

import sys
import types

import kornia.utils

if "kornia.utils.grid" not in sys.modules:
    _grid = types.ModuleType("kornia.utils.grid")
    for _name in ("create_meshgrid", "create_meshgrid3d"):
        _fn = getattr(kornia.utils, _name, None)
        if _fn is not None:
            setattr(_grid, _name, _fn)
    sys.modules["kornia.utils.grid"] = _grid
    kornia.utils.grid = _grid
