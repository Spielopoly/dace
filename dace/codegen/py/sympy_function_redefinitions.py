import numpy as _np

Min = min
Max = max
Abs = abs


# Integer-exact ceil/floor division: NO float cast (cuTile cannot int() tiles;
# see commit a18642205).
def int_ceil(x, y=1):
    return -(-x // y)


def int_floor(x, y=1):
    return x // y


# Sympy boolean if-then-else (emitted verbatim by boundary-ITE tasklets, e.g.
# ``_o = ITE(cond, _new, _old)``). A plain-bool condition branches directly
# (cupy.where rejects scalar conditions); array conditions go through
# ``np.where`` (dispatches to cupy via __array_function__).
def ITE(cond, then_value, else_value):
    if isinstance(cond, (bool, _np.bool_)):
        return then_value if cond else else_value
    return _np.where(cond, then_value, else_value)


# Math functions emitted verbatim by DaCe tasklets (e.g. ``np.sqrt`` lowered to
# ``__out = sqrt(__in1)``). The canonical name set comes from the tasklet
# bodies in ``dace/frontend/python/replacements/`` (mostly ``ufunc.py``) and
# the C++ runtime's ``dace/runtime/include/dace/math.h``, which the C++
# backend resolves to ``dace::math::*``. For the Python backend each name maps
# to its NumPy equivalent; numpy dispatches on the operand type, so NumPy and
# CuPy scalars both work.
_NUMPY_EQUIVALENTS = {
    # NumPy-named elementwise functions.
    'sqrt': 'sqrt',
    'cbrt': 'cbrt',
    'exp': 'exp',
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log': 'log',
    'log2': 'log2',
    'log10': 'log10',
    'log1p': 'log1p',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
    'arcsin': 'arcsin',
    'arccos': 'arccos',
    'arctan': 'arctan',
    'arctan2': 'arctan2',
    'arcsinh': 'arcsinh',
    'arccosh': 'arccosh',
    'arctanh': 'arctanh',
    'floor': 'floor',
    'ceil': 'ceil',
    'trunc': 'trunc',
    'round': 'round',
    'fabs': 'fabs',
    'sign': 'sign',
    'conj': 'conj',
    'real': 'real',
    'imag': 'imag',
    'fmin': 'fmin',
    'fmax': 'fmax',
    'fmod': 'fmod',
    'hypot': 'hypot',
    'copysign': 'copysign',
    'ldexp': 'ldexp',
    'nextafter': 'nextafter',
    'heaviside': 'heaviside',
    'deg2rad': 'deg2rad',
    'rad2deg': 'rad2deg',
    'reciprocal': 'reciprocal',
    'gcd': 'gcd',
    'lcm': 'lcm',
    'isfinite': 'isfinite',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'signbit': 'signbit',
    # C / sympy spellings.
    'asin': 'arcsin',
    'acos': 'arccos',
    'atan': 'arctan',
    'atan2': 'arctan2',
    'asinh': 'arcsinh',
    'acosh': 'arccosh',
    'atanh': 'arctanh',
    'pow': 'power',
    'ceiling': 'ceil',
    # DaCe runtime helper names (dace/runtime/include/dace/math.h).
    'py_mod': 'mod',  # Python-sign modulo
    'cpp_mod': 'fmod',  # C-sign modulo
    'py_floor': 'floor_divide',  # floor division
    'np_float_pow': 'float_power',
    'sign_numpy_2': 'sign',
}
for _alias, _np_name in _NUMPY_EQUIVALENTS.items():
    globals()[_alias] = getattr(_np, _np_name)
