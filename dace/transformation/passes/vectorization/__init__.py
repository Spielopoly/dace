"""Public vectorization passes: the CPU map/tasklet vectorizers and the
cuTile lowering pipeline."""
from .vectorize import Vectorize
from .vectorize_cpu import VectorizeCPU
from .vectorize_cutile import VectorizeCuTile
# Importing this module registers the ``"vectorized"`` implementation on
# the standard ``Reduce`` library node (schedule-aware dispatcher).
from . import reduce_expansion  # noqa: F401
