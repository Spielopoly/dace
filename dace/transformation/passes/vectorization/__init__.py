# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Public vectorization passes: the multi-dim CPU/GPU tile-op vectorizer.

The pipeline entry points (``VectorizeMultiDim`` / ``VectorizeCPUMultiDim`` /
``VectorizeGPUMultiDim`` / ``VectorizeCuTile``) are exported LAZILY via :pep:`562`
``__getattr__``. ``vectorize_multi_dim`` imports ``dace.transformation.interstate``
at module load, and ``interstate`` in turn (through ``passes -> canonicalize``)
imports back into this package -- eager-importing the pipeline here would close
that cycle. Deferring it means importing a vectorization *submodule* does not drag
in the whole pipeline, while ``from ...vectorization import VectorizeCPUMultiDim``
still works on demand.
"""
import importlib

# Importing this module registers the ``"vectorized"`` implementation on the
# standard ``Reduce`` library node (schedule-aware dispatcher). Cycle-safe.
from . import reduce_expansion  # noqa: F401

#: Lazily-exported pipeline entry points, mapped to their defining submodule.
_PIPELINE_EXPORTS = {
    "VectorizeMultiDim": "vectorize_multi_dim",
    "VectorizeCPUMultiDim": "vectorize_multi_dim",
    "VectorizeGPUMultiDim": "vectorize_multi_dim",
    "VectorizeCuTile": "vectorize_cutile",
}


def __getattr__(name):
    """Lazily resolve the pipeline entry points (breaks the interstate import cycle)."""
    if name in _PIPELINE_EXPORTS:
        module = importlib.import_module(f"{__name__}.{_PIPELINE_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_PIPELINE_EXPORTS))
