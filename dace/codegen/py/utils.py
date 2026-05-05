from dace import data, subsets

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.codegen.py.target import PythonTargetCodeGenerator


def numpy_array_expression(sdfg,
                   memlet,
                   with_brackets=True,
                   offset=None,
                   relative_offset=True,
                   packed_veclen=1,
                   use_other_subset=False,
                   indices=None,
                   referenced_array=None,
                   codegen: 'PythonTargetCodeGenerator | None' = None,
                   framecode: 'DaCePythonCodeGenerator | None' = None):
    """ Converts an Indices/Range object to a numpy array access string. """
    # TODO: Make python compatible, update docstring
    subset = memlet.subset if not use_other_subset else memlet.other_subset
    s = subset if relative_offset else subsets.Range.from_indices(offset)
    o = offset if relative_offset else None
    desc = (sdfg.arrays[memlet.data] if referenced_array is None else referenced_array)
    offset_str = numpy_offset_expression(desc, s, o, packed_veclen, indices=indices)

    name = memlet.data

    if with_brackets:
        if codegen is not None:
            ptrname = codegen.ptr(name, desc, sdfg, memlet.subset)
        else:
            ptrname = ptr(name, desc, sdfg, framecode=framecode)
        return "%s[%s]" % (ptrname, offset_str)
    else:
        return offset_str

def numpy_offset_expression(d: data.Data, subset_in: subsets.Subset, offset=None, packed_veclen=1, indices=None) -> str:
    """ Creates a C++ expression that can be added to a pointer in order
        to offset it to the beginning of the given subset and offset.

        :param d: The data structure to use for sizes/strides.
        :param subset_in: The subset to offset by.
        :param offset: An additional list of offsets or a Subset object
        :param packed_veclen: If packed types are targeted, specifies the
                              vector length that the final offset should be
                              divided by.
        :param indices: A tuple of indices to use for expression.
        :param codegen: Optional code generator to adjust subset.
        :return: A string in C++ syntax with the correct offset
    """
    # TODO: Make python compatible, update docstring
    # Offset according to parameters, then offset according to array
    if offset is not None:
        subset = subset_in.offset_new(offset, False)
        subset.offset(d.offset, False)
    else:
        subset = subset_in.offset_new(d.offset, False)

    # Obtain start range from offsetted subset
    indices = indices or ([0] * len(d.strides))

    index = subset.at(indices, d.strides)
    if packed_veclen > 1:
        index /= packed_veclen

    return sym2cpp(index)