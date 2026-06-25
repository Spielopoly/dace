"""Shared DaCe-to-cuTile dtype mapping for tileops library nodes."""

import dace


_DACE_TO_CUTILE_DTYPE = {
    "float16": "ct.float16",
    "float32": "ct.float32",
    "float64": "ct.float64",
    "int8": "ct.int8",
    "int16": "ct.int16",
    "int32": "ct.int32",
    "int64": "ct.int64",
    "uint8": "ct.uint8",
    "uint16": "ct.uint16",
    "uint32": "ct.uint32",
    "uint64": "ct.uint64",
    "bool": "ct.bool_",
}


def dace_dtype_to_cutile_str(dtype: "dace.dtypes.typeclass") -> str:
    """Map a DaCe dtype to a ``ct.<type>`` expression string for cuTile codegen.

    :param dtype: A ``dace.dtypes.typeclass`` instance (e.g. ``dace.float64``).
    :returns: A string like ``"ct.float64"`` suitable for embedding in
        generated cuTile Python code.
    :raises ValueError: If the dtype has no known cuTile equivalent.
    """
    key = dtype.to_string()
    try:
        return _DACE_TO_CUTILE_DTYPE[key]
    except KeyError:
        raise ValueError(
            f"No cuTile dtype mapping for DaCe dtype {dtype!r} "
            f"(to_string={key!r})"
        )
