"""Runtime alias for the pre-0.4 sweep-manager module path.

The module object itself is replaced, rather than merely re-exporting names,
so patches applied through ``pynst.sweep_manager`` affect the globals used by
``pynst.execution.manager.SweepManager``.
"""

from __future__ import annotations

import sys as _sys

from .execution import manager as _implementation


_sys.modules[__name__] = _implementation
