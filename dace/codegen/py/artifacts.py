# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Typed build artifacts for the Python-backend cuTile target."""

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple, Type

from dace.codegen.codeobject import CodeObject


class CuTileParameterKind(Enum):
    """Storage kinds supported by the exported cuTile ABI."""

    Array = "array"
    ReadOnlyScalar = "read_only_scalar"
    MutableScalar = "mutable_scalar"


@dataclass(frozen=True)
class CuTileParameterSpec:
    """One logical argument of an exported cuTile kernel.

    :param name: Name used in the generated kernel branch.
    :param host_expression: Expression passed by the generated host program.
    :param kind: Array or scalar storage kind.
    :param dace_dtype: Canonical NumPy spelling of the DaCe data type.
    :param cutile_dtype: Attribute name on the ``cuda.tile`` module.
    :param rank: Array rank, or zero for a read-only scalar.
    :param index_dtype: Attribute name of the array index type.
    :param shape_constraints: Per-dimension constant shapes, when proven.
    :param stride_constraints: Per-dimension constant element strides.
    :param alias_groups: Conservative alias groups for array parameters.
    :param may_alias_internally: Whether distinct indices may alias.
    :param is_read: Whether the kernel reads the parameter.
    :param is_written: Whether the kernel writes the parameter.
    :param source: SDFG descriptor or symbol name.
    """

    name: str
    host_expression: str
    kind: CuTileParameterKind
    dace_dtype: str
    cutile_dtype: str
    rank: int
    index_dtype: str
    shape_constraints: Tuple[Optional[int], ...]
    stride_constraints: Tuple[Optional[int], ...]
    alias_groups: Tuple[str, ...]
    may_alias_internally: bool
    is_read: bool
    is_written: bool
    source: str

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Invalid cuTile parameter name {self.name!r}")
        if self.kind == CuTileParameterKind.ReadOnlyScalar:
            if self.rank != 0 or self.shape_constraints or self.stride_constraints:
                raise ValueError("Read-only scalar parameters must have rank zero")
            if self.is_written:
                raise ValueError("A read-only scalar parameter cannot be written")
        elif self.kind in (CuTileParameterKind.Array, CuTileParameterKind.MutableScalar):
            if self.rank < 1:
                raise ValueError("Array ABI parameters must have positive rank")
            if len(self.shape_constraints) != self.rank or len(self.stride_constraints) != self.rank:
                raise ValueError("Array constraints must match the parameter rank")
        if self.index_dtype not in ("int32", "int64"):
            raise ValueError(f"Unsupported cuTile index type {self.index_dtype!r}")

    def render_constraint(self) -> str:
        """Render this parameter's explicit cuTile constraint.

        :returns: Python source for an ``ArrayConstraint`` or
            ``ScalarConstraint``.
        """
        if self.kind == CuTileParameterKind.ReadOnlyScalar:
            return f"compilation.ScalarConstraint(ct.{self.cutile_dtype})"

        shapes = _optional_int_tuple(self.shape_constraints)
        strides = _optional_int_tuple(self.stride_constraints)
        aliases = repr(self.alias_groups)
        return (f"compilation.ArrayConstraint(ct.{self.cutile_dtype}, {self.rank}, "
                f"index_dtype=ct.{self.index_dtype}, "
                "stride_lower_bound_incl=0, "
                f"alias_groups={aliases}, may_alias_internally={self.may_alias_internally!r}, "
                f"stride_constant={strides}, shape_constant={shapes})")

    def abi_dict(self) -> dict:
        """Return the deterministic ABI fields used by content hashing."""
        result = dataclasses.asdict(self)
        result["kind"] = self.kind.value
        return result


@dataclass(frozen=True)
class CuTileKernelSpec:
    """Typed source and ABI for one cuTile map scope."""

    stable_identity: Tuple[int, int, int, str]
    map_identity: str
    source_location: str
    branch_body: str
    helper_source: str
    parameters: Tuple[CuTileParameterSpec, ...]
    exported_symbol: str
    launch_helper: str
    grid: Tuple[str, ...]
    cuda_block: Tuple[int, int, int] = (1, 1, 1)
    tile_widths: Tuple[str, ...] = ()
    compile_time_values: Tuple[Tuple[str, str], ...] = ()
    required_imports: Tuple[str, ...] = ()
    compiler_options: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.exported_symbol.isidentifier():
            raise ValueError(f"Invalid exported cuTile symbol {self.exported_symbol!r}")
        if not self.launch_helper.isidentifier():
            raise ValueError(f"Invalid cuTile launch helper {self.launch_helper!r}")
        if not 1 <= len(self.grid) <= 3:
            raise ValueError("A cuTile launch grid must have one to three dimensions")
        if self.cuda_block != (1, 1, 1):
            raise ValueError("Exported cuTile kernels require CUDA block (1, 1, 1)")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate cuTile parameter in {self.map_identity}")

    @property
    def content_hash(self) -> str:
        """Return a short deterministic hash of the branch body and ABI."""
        content = {
            "body": self.branch_body,
            "parameters": [parameter.abi_dict() for parameter in self.parameters],
            "grid": self.grid,
            "tile_widths": self.tile_widths,
            "compile_time_values": self.compile_time_values,
        }
        payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def render_launch_helper(self) -> str:
        """Render the dedicated Cython launch helper for this map.

        :returns: Cython source containing fixed validation, argument packing,
            grid calculation, symbol lookup, and launch code.
        """
        lines = [f"cdef void {self.launch_helper}("]
        for i in range(len(self.parameters)):
            suffix = "," if i + 1 < len(self.parameters) else ""
            lines.append(f"        object __dace_value_{i}{suffix}")
        lines.append(") except *:")
        declarations = [
            "cdef list __dace_args = []",
            "cdef object __dace_function",
            "cdef object __dace_stream",
            "cdef tuple __dace_grid",
        ]
        for i, parameter in enumerate(self.parameters):
            declarations.append(f"cdef object __dace_packed_{i}")
            if parameter.kind != CuTileParameterKind.ReadOnlyScalar:
                declarations.extend([
                    f"cdef object __dace_iface_{i}",
                    f"cdef long long __dace_itemsize_{i}",
                    f"cdef long long __dace_byte_stride_{i}",
                ])
        lines.extend(f"    {line}" for line in declarations)

        for i, parameter in enumerate(self.parameters):
            if parameter.kind == CuTileParameterKind.ReadOnlyScalar:
                lines.extend(_render_scalar_pack(i, parameter))
            else:
                lines.extend(_render_array_pack(i, parameter))

        helper_grid = []
        for expression in self.grid:
            for i, parameter in enumerate(self.parameters):
                expression = re.sub(rf"\b{re.escape(parameter.name)}\b", f"__dace_value_{i}", expression)
            helper_grid.append(expression)
        grid_values = ", ".join(f"int({expression})" for expression in helper_grid)
        if len(self.grid) == 1:
            grid_values += ","
        lines.append(f"    __dace_grid = ({grid_values})")
        lines.append("    if min(__dace_grid) <= 0:")
        lines.append("        return")
        lines.append(f"    __dace_function = __dace_cutile_get_function({self.exported_symbol!r})")
        lines.append("    __dace_stream = cupy.cuda.get_current_stream()")
        padded_grid = ", ".join(f"__dace_grid[{i}]" if i < len(self.grid) else "1" for i in range(3))
        lines.extend((
            "    __dace_function(",
            f"        ({padded_grid}),",
            f"        {self.cuda_block!r},",
            "        tuple(__dace_args),",
            "        stream=__dace_stream)",
        ))
        return "\n".join(lines) + "\n"

    def render_host_call(self) -> str:
        """Render the call from the generated SDFG body to this helper."""
        if not self.parameters:
            return f"{self.launch_helper}()"
        arguments = ",\n".join(f"    {parameter.host_expression}" for parameter in self.parameters)
        return f"{self.launch_helper}(\n{arguments})"


@dataclass(frozen=True)
class CuTileModuleSpec:
    """Aggregate build-only cuTile module for one top-level SDFG target."""

    name: str
    kernels: Tuple[CuTileKernelSpec, ...] = ()

    def with_kernel(self, kernel: CuTileKernelSpec) -> "CuTileModuleSpec":
        """Return a new module specification containing ``kernel``."""
        if any(existing.stable_identity == kernel.stable_identity for existing in self.kernels):
            raise ValueError(f"Duplicate cuTile map identity {kernel.map_identity}")
        return dataclasses.replace(self, kernels=self.kernels + (kernel, ))

    @property
    def ordered_kernels(self) -> Tuple[CuTileKernelSpec, ...]:
        """Return kernels sorted by stable graph identity."""
        return tuple(sorted(self.kernels, key=lambda kernel: kernel.stable_identity))

    @property
    def symbol_map(self) -> dict:
        """Return deterministic exported-symbol to map-identity metadata."""
        return {kernel.exported_symbol: kernel.map_identity for kernel in self.ordered_kernels}

    def render_source(self) -> str:
        """Render the executable aggregate cuTile build source."""
        kernels = self.ordered_kernels
        if not kernels:
            return ""

        imports = {
            "import argparse",
            "import cuda.tile as ct",
            "from cuda.tile import compilation",
            "from pathlib import Path",
        }
        for kernel in kernels:
            imports.update(kernel.required_imports)
        lines = sorted(imports) + [""]
        lines.extend((
            "Min = min",
            "Max = max",
            "Abs = abs",
            "",
            "def int_ceil(x, y):",
            "    return -(-x // y)",
            "",
            "def int_floor(x, y):",
            "    return x // y",
            "",
            "def py_mod(x, y):",
            "    return x % y",
            "",
        ))

        helper_sources = sorted({kernel.helper_source.strip() for kernel in kernels if kernel.helper_source.strip()})
        for helper in helper_sources:
            lines.extend((helper, ""))

        lines.extend(("@ct.kernel", "def __dace_cutile_module(kernel_id: ct.Constant[int], args):"))
        for branch_id, kernel in enumerate(kernels):
            keyword = "if" if branch_id == 0 else "elif"
            lines.append(f"    {keyword} kernel_id == {branch_id}:")
            parameter_names = ", ".join(parameter.name for parameter in kernel.parameters)
            if len(kernel.parameters) == 1:
                parameter_names += ","
            if kernel.parameters:
                lines.append(f"        {parameter_names} = args")
            body = kernel.branch_body.rstrip()
            if body:
                lines.extend(f"        {line}" if line else "" for line in body.splitlines())
            else:
                lines.append("        pass")

        lines.extend(("", "__dace_cutile_signatures = ["))
        for branch_id, kernel in enumerate(kernels):
            constraints = ",\n".join(f"                {parameter.render_constraint()}"
                                     for parameter in kernel.parameters)
            if constraints:
                constraints += ","
            lines.extend((
                "    compilation.KernelSignature(",
                "        [",
                f"            compilation.ConstantConstraint({branch_id}),",
                "            compilation.TupleConstraint((",
                constraints,
                "            )),",
                "        ],",
                "        compilation.CallingConvention.cutile_python_v2(),",
                f"        symbol={kernel.exported_symbol!r},",
                "    ),",
            ))
        lines.extend((
            "]",
            "",
            "def export(output_path, target_arch):",
            "    compilation.export_kernel(",
            "        __dace_cutile_module,",
            "        __dace_cutile_signatures,",
            "        Path(output_path),",
            "        gpu_code=target_arch,",
            "        output_format='cubin',",
            "    )",
            "",
            "def _main():",
            "    parser = argparse.ArgumentParser()",
            "    parser.add_argument('--output', required=True)",
            "    parser.add_argument('--arch', required=True)",
            "    args = parser.parse_args()",
            "    export(args.output, args.arch)",
            "",
            "if __name__ == '__main__':",
            "    _main()",
            "",
        ))
        return "\n".join(lines)

    def render_codeobject(self, target: Type[object]) -> Optional[CodeObject]:
        """Return the one build-only code object, or ``None`` when empty."""
        if not self.kernels:
            return None
        return CodeObject(
            name=f"{self.name}_cutile_build",
            code=self.render_source(),
            language="py",
            target=target,
            title="cuTile aggregate build module",
            target_type="cutile_build",
            additional_compiler_kwargs={"cutile_symbols": json.dumps(self.symbol_map, sort_keys=True)},
            linkable=False,
        )


def make_kernel_symbol(stable_identity: Tuple[int, int, int, str], body: str,
                       parameters: Iterable[CuTileParameterSpec]) -> Tuple[str, str]:
    """Return deterministic exported-symbol and launch-helper names."""
    cfg_id, state_id, map_id, _ = stable_identity
    content = {
        "body": body,
        "parameters": [parameter.abi_dict() for parameter in parameters],
    }
    payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    symbol = f"__dace_cutile_map_c{cfg_id}_s{state_id}_m{map_id}_{digest}"
    helper = f"__dace_cutile_launch_c{cfg_id}_s{state_id}_m{map_id}_{digest}"
    return symbol, helper


def render_host_support() -> str:
    """Render shared embedded-cubin declarations and per-context caches."""
    return """cdef extern from \"dace_cutile_embedded.h\":
    const unsigned char __dace_cutile_cubin[]
    const size_t __dace_cutile_cubin_size

cdef dict __dace_cutile_module_cache = {}
cdef dict __dace_cutile_function_cache = {}

cdef tuple __dace_cutile_context_key():
    cdef int device = cupy.cuda.Device().id
    cdef long long context = int(cupy.cuda.driver.ctxGetCurrent())
    if context == 0:
        raise RuntimeError(\"A current CUDA context is required for a cuTile launch\")
    return device, context

cdef object __dace_cutile_get_function(str symbol):
    cdef tuple context_key = __dace_cutile_context_key()
    cdef tuple function_key = (context_key, symbol)
    cdef object module = __dace_cutile_module_cache.get(context_key)
    cdef object function = __dace_cutile_function_cache.get(function_key)
    if function is not None:
        return function
    if module is None:
        if __dace_cutile_cubin_size == 0:
            raise RuntimeError(\"The compiled SDFG contains an empty cuTile cubin\")
        module = cupy.cuda.function.Module()
        module.load(PyBytes_FromStringAndSize(
            <const char *>__dace_cutile_cubin, __dace_cutile_cubin_size))
        __dace_cutile_module_cache[context_key] = module
    function = module.get_function(symbol)
    __dace_cutile_function_cache[function_key] = function
    return function
"""


def _optional_int_tuple(values: Tuple[Optional[int], ...]) -> str:
    return "(" + ", ".join("None" if value is None else str(value)
                           for value in values) + ("," if len(values) == 1 else "") + ")"


def _render_array_pack(index: int, parameter: CuTileParameterSpec) -> list:
    value = f"__dace_value_{index}"
    iface = f"__dace_iface_{index}"
    lines = [
        f"    {iface} = getattr({value}, '__cuda_array_interface__', None)",
        f"    if {iface} is None:",
        f"        raise TypeError({('cuTile parameter ' + parameter.source + ' must be a CUDA array')!r})",
        f"    if {value}.ndim != {parameter.rank}:",
        f"        raise TypeError(f\"cuTile parameter {parameter.source} must have rank {parameter.rank}, got {{{value}.ndim}}\")",
        f"    if numpy.dtype({value}.dtype) != numpy.dtype({parameter.dace_dtype!r}):",
        f"        raise TypeError(f\"cuTile parameter {parameter.source} must have dtype {parameter.dace_dtype}, got {{{value}.dtype}}\")",
        f"    __dace_itemsize_{index} = int({value}.dtype.itemsize)",
        f"    if __dace_itemsize_{index} <= 0:",
        f"        raise ValueError({('cuTile parameter ' + parameter.source + ' has an invalid item size')!r})",
        f"    __dace_args.append({value})",
    ]
    for dim, constant in enumerate(parameter.shape_constraints):
        lines.append(f"    if int({value}.shape[{dim}]) < 0 or int({value}.shape[{dim}]) > 9223372036854775807:")
        lines.append(
            f"        raise OverflowError(f\"shape {dim} of cuTile parameter {parameter.source} does not fit int64\")")
        if constant is not None:
            lines.append(f"    if int({value}.shape[{dim}]) != {constant}:")
            lines.append(
                f"        raise ValueError(f\"shape {dim} of cuTile parameter {parameter.source} must be {constant}\")")
        lines.append(f"    __dace_args.append(numpy.int64({value}.shape[{dim}]))")
    for dim, constant in enumerate(parameter.stride_constraints):
        trailing_shape = " * ".join(f"int({value}.shape[{d}])" for d in range(dim + 1, parameter.rank)) or "1"
        lines.extend((
            f"    __dace_byte_stride_{index} = (",
            f"        __dace_itemsize_{index} * ({trailing_shape}) if {value}.strides is None else",
            f"        int({value}.strides[{dim}]))",
        ))
        lines.append(
            f"    if __dace_byte_stride_{index} < 0 or __dace_byte_stride_{index} % __dace_itemsize_{index} != 0:")
        lines.append(f"        raise ValueError(f\"cuTile parameter {parameter.source} has an unsupported byte stride "
                     f"{{__dace_byte_stride_{index}}} in dimension {dim}\")")
        lines.append(f"    if __dace_byte_stride_{index} // __dace_itemsize_{index} > 9223372036854775807:")
        lines.append(
            f"        raise OverflowError(f\"stride {dim} of cuTile parameter {parameter.source} does not fit int64\")")
        if constant is not None:
            lines.append(f"    if __dace_byte_stride_{index} // __dace_itemsize_{index} != {constant}:")
            lines.append(
                f"        raise ValueError(f\"stride {dim} of cuTile parameter {parameter.source} must be {constant}\")"
            )
        lines.append(f"    __dace_args.append(numpy.int64(__dace_byte_stride_{index} // __dace_itemsize_{index}))")
    return lines


def _render_scalar_pack(index: int, parameter: CuTileParameterSpec) -> list:
    value = f"__dace_value_{index}"
    packed = f"__dace_packed_{index}"
    raw = f"__dace_raw_{index}"
    lines = [
        f"    {raw} = {value}.item() if hasattr({value}, 'item') else {value}",
    ]
    dtype = parameter.dace_dtype
    numpy_constructor = "bool_" if dtype == "bool" else dtype
    if dtype == "bool":
        lines.extend((
            f"    if not isinstance({raw}, (bool, numpy.bool_)):",
            f"        raise TypeError(f\"cuTile scalar {parameter.source} must be bool, got {{type({raw}).__name__}}\")",
        ))
    elif dtype.startswith("int") or dtype.startswith("uint"):
        bits = int(dtype[4:] if dtype.startswith("uint") else dtype[3:])
        if dtype.startswith("uint"):
            lower, upper = 0, 2**bits - 1
        else:
            lower, upper = -(2**(bits - 1)), 2**(bits - 1) - 1
        lines.extend((
            f"    if isinstance({raw}, (bool, numpy.bool_)) or not isinstance({raw}, (int, numpy.integer)):",
            f"        raise TypeError(f\"cuTile scalar {parameter.source} must be an integer, got {{type({raw}).__name__}}\")",
            f"    if {raw} < {lower} or {raw} > {upper}:",
            f"        raise OverflowError({('cuTile scalar ' + parameter.source + ' is outside ' + dtype + ' range')!r})",
        ))
    elif dtype.startswith("float"):
        lines.extend((
            f"    if isinstance({raw}, (bool, numpy.bool_)) or not isinstance("
            f"{raw}, (int, float, numpy.integer, numpy.floating)):",
            f"        raise TypeError(f\"cuTile scalar {parameter.source} must be real, got {{type({raw}).__name__}}\")",
            f"    if isinstance({raw}, (int, numpy.integer)) and "
            f"abs({raw}) > int(numpy.finfo(numpy.{dtype}).max):",
            f"        raise OverflowError({('cuTile scalar ' + parameter.source + ' is outside ' + dtype + ' range')!r})",
            f"    if isinstance({raw}, (float, numpy.floating)) and numpy.isfinite({raw}) and abs({raw}) > numpy.finfo(numpy.{dtype}).max:",
            f"        raise OverflowError({('cuTile scalar ' + parameter.source + ' is outside ' + dtype + ' range')!r})",
        ))
    else:
        raise ValueError(f"Unsupported read-only cuTile scalar type {dtype!r}")
    lines.extend((
        f"    {packed} = numpy.{numpy_constructor}({raw})",
        f"    __dace_args.append({packed})",
    ))
    return lines
