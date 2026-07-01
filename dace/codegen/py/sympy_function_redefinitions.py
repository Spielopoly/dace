import numpy as _np

Min = min
Max = max
Abs = abs
def int_ceil(x, y=1):
    return -(-x // y)
def int_floor(x, y=1):
    return x // y

# Elementwise math functions emitted verbatim by DaCe tasklets (e.g. a
# ``np.sqrt`` reduction lowered to ``__out = sqrt(__in1)``).  The C++ backend
# resolves these to ``dace::math::*``; for the Python backend they must be
# plain callables.  ``numpy`` provides host/device-agnostic implementations
# (they dispatch on the operand type, so a NumPy or CuPy scalar both work).
sqrt = _np.sqrt
exp = _np.exp
log = _np.log
sin = _np.sin
cos = _np.cos
tan = _np.tan
tanh = _np.tanh
sinh = _np.sinh
cosh = _np.cosh
arcsin = _np.arcsin
arccos = _np.arccos
arctan = _np.arctan
floor = _np.floor
ceil = _np.ceil
fabs = _np.fabs
sign = _np.sign
log10 = _np.log10
log2 = _np.log2
conj = _np.conj
real = _np.real
imag = _np.imag
