# ------------------------------------------------------------------
# Expose inner sub-packages *and* modules under the public namespace.
#  - vortex.vortex.model    → vortex.model
#  - vortex.vortex.logging  → vortex.logging
#  - vortex.vortex.ops      → vortex.ops
# ------------------------------------------------------------------
import importlib, pkgutil, sys, pathlib

_inner_root = pathlib.Path(__file__).parent / "vortex"   # …/vortex/vortex
_prefix     = f"{__name__}.vortex."

for mod_info in pkgutil.iter_modules([str(_inner_root)], prefix=_prefix):
    imported = importlib.import_module(mod_info.name)          # import once
    alias    = f"{__name__}.{mod_info.name.split('.')[-1]}"    # short key
    sys.modules[alias] = imported                              # register

del importlib, pkgutil, sys, pathlib, _inner_root, _prefix, mod_info, imported, alias
