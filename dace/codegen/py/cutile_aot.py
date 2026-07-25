# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ahead-of-time compilation support for generated cuTile kernels."""
import base64
import io
import linecache
import pprint
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dace.config import Config

AOT_ABI_VERSION = 1


class CuTileAOTError(RuntimeError):
    """Raised when a cuTile kernel cannot be exported or launched."""


@dataclass(frozen=True)
class AOTParam:
    """One parameter in the generated direct-launch ABI."""

    kind: str
    dtype: str
    ndim: int
    strides: Optional[Tuple[Optional[int], ...]] = None

    def to_metadata(self) -> Dict[str, Any]:
        """Return the self-contained serialized representation."""
        return {
            'kind': self.kind,
            'dtype': self.dtype,
            'ndim': self.ndim,
            'strides': self.strides,
        }


def _compilation() -> Any:
    try:
        from cuda.tile import compilation
    except ImportError as exc:
        raise CuTileAOTError(f'cuda-tile is required for cuTile AOT compilation: {exc}') from exc
    return compilation


def check_prerequisites() -> None:
    """Check the public cuda-tile APIs required by the AOT path."""
    compilation = _compilation()
    required = ('export_kernel', 'KernelSignature', 'ArrayConstraint', 'ScalarConstraint', 'CallingConvention')
    missing = [name for name in required if not hasattr(compilation, name)]
    if missing or not hasattr(compilation.CallingConvention, 'cutile_python_v2'):
        raise CuTileAOTError(f'cuda.tile.compilation lacks required public APIs: {missing or ["cutile_python_v2"]}')


def resolve_arch() -> str:
    """Resolve the architecture used for cubin export."""
    configured = Config.get('compiler', 'cutile', 'aot_arch')
    if configured:
        return str(configured)
    try:
        import cupy
        return f'sm_{cupy.cuda.Device().compute_capability}'
    except Exception as exc:
        raise CuTileAOTError('Cannot determine target GPU architecture; set compiler.cutile.aot_arch') from exc


def _ct_dtype(name: str) -> Any:
    import cuda.tile as ct
    result = getattr(ct, 'bool_' if name == 'bool' else name, None)
    if result is None:
        raise CuTileAOTError(f'cuda.tile has no dtype matching {name!r}')
    return result


def _validated_params(spec: Mapping[str, Any]) -> Tuple[AOTParam, ...]:
    """Validate and normalize an AOT kernel signature descriptor."""
    expected_spec_fields = {'abi_version', 'params'}
    if set(spec) != expected_spec_fields:
        raise CuTileAOTError(f'Invalid cuTile AOT descriptor fields {set(spec)!r}; '
                             f'expected {expected_spec_fields!r}')
    if spec['abi_version'] != AOT_ABI_VERSION:
        raise CuTileAOTError(f"Unsupported cuTile AOT ABI version {spec['abi_version']!r}; "
                             f'expected {AOT_ABI_VERSION}')

    result: List[AOTParam] = []
    for value in spec['params']:
        if isinstance(value, AOTParam):
            param = value
        elif isinstance(value, Mapping):
            expected_param_fields = {'kind', 'dtype', 'ndim', 'strides'}
            if set(value) != expected_param_fields:
                raise CuTileAOTError(f'Invalid cuTile AOT parameter fields {set(value)!r}; '
                                     f'expected {expected_param_fields!r}')
            strides = value['strides']
            param = AOTParam(value['kind'], value['dtype'], value['ndim'], None if strides is None else tuple(strides))
        else:
            raise CuTileAOTError(f'Invalid cuTile AOT parameter descriptor {value!r}')

        if param.kind not in ('array', 'scalar') or not isinstance(param.dtype, str):
            raise CuTileAOTError(f'Invalid cuTile AOT parameter descriptor {param!r}')
        if type(param.ndim) is not int or param.ndim < 0:
            raise CuTileAOTError(f'Invalid cuTile AOT parameter rank in {param!r}')
        if param.kind == 'scalar':
            if param.ndim != 0 or param.strides is not None:
                raise CuTileAOTError(f'Scalar cuTile AOT parameter must have rank zero and no strides: {param!r}')
        elif param.ndim == 0:
            raise CuTileAOTError(f'Array cuTile AOT parameter must have positive rank: {param!r}')
        elif param.strides is not None:
            if len(param.strides) != param.ndim or any(stride is not None and (type(stride) is not int or stride < 0)
                                                       for stride in param.strides):
                raise CuTileAOTError(f'Invalid cuTile AOT stride constraints in {param!r}')
        result.append(param)
    return tuple(result)


def build_export_plan(spec: Mapping[str, Any], function_name: str) -> Tuple[List[Any], List[Dict[str, str]]]:
    """Build conservative v2 signatures for the required index widths."""
    compilation = _compilation()
    parameters = _validated_params(spec)
    alias_groups: Tuple[str, ...] = ('dace', ) if sum(p.kind == 'array' for p in parameters) > 1 else ()
    signatures, variants = [], []
    index_dtype_names = ('int32', 'int64') if any(p.kind == 'array' for p in parameters) else ('int32', )
    for index_dtype_name in index_dtype_names:
        constraints = []
        for param in parameters:
            if param.kind == 'array':
                constraints.append(
                    compilation.ArrayConstraint(_ct_dtype(param.dtype),
                                                param.ndim,
                                                index_dtype=_ct_dtype(index_dtype_name),
                                                stride_lower_bound_incl=0,
                                                alias_groups=alias_groups,
                                                may_alias_internally=True,
                                                stride_constant=param.strides))
            elif param.kind == 'scalar':
                constraints.append(compilation.ScalarConstraint(_ct_dtype(param.dtype)))
            else:
                raise CuTileAOTError(f'Unsupported AOT parameter kind {param.kind!r}')
        signature = compilation.KernelSignature(
            constraints, compilation.CallingConvention.cutile_python_v2()).with_mangled_symbol(function_name)
        signatures.append(signature)
        variants.append({'index_dtype': index_dtype_name, 'symbol': signature.symbol})
    return signatures, variants


def _load_kernel(source: str, function_name: str, namespace: Dict[str, Any]) -> Any:
    filename = f'<dace_cutile_aot_{function_name}>'
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        exec(compile(source, filename, 'exec'), namespace)
        return namespace[function_name]
    except Exception as exc:
        raise CuTileAOTError(f'Cannot construct generated cuTile kernel {function_name}: {exc}') from exc


def _export_kernel(kernel: Any, signatures: Sequence[Any], arch: str, function_name: str) -> bytes:
    output = io.BytesIO()
    try:
        _compilation().export_kernel(kernel, signatures, output, gpu_code=arch, output_format='cubin')
    except Exception as exc:
        raise CuTileAOTError(f'AOT export failed for kernel {function_name!r} ({arch}): {exc}') from exc
    if not output.getvalue():
        raise CuTileAOTError(f'AOT export produced an empty cubin for {function_name!r}')
    return output.getvalue()


def generate_aot_module(kernels: Sequence[Dict[str, Any]], module_name: str) -> str:
    """Export collected kernels and return an embedded direct-launch module."""
    check_prerequisites()
    arch = resolve_arch()
    import cuda.tile as ct
    namespace: Dict[str, Any] = {'ct': ct, '__name__': module_name}
    metadata: Dict[str, Any] = {}
    for descriptor in kernels:
        name = descriptor['name']
        spec = {
            'abi_version': descriptor.get('abi_version'),
            'params': descriptor.get('params'),
        }
        params = _validated_params(spec)
        filename = f'<dace_cutile_aot_{name}>'
        try:
            kernel = _load_kernel(descriptor['source'], name, namespace)
            signatures, variants = build_export_plan(spec, name)
            cubin = _export_kernel(kernel, signatures, arch, name)
        finally:
            linecache.cache.pop(filename, None)
        metadata[name] = {
            'abi_version': AOT_ABI_VERSION,
            'arch': arch,
            'cubin': base64.b85encode(zlib.compress(cubin, level=9)).decode('ascii'),
            'params': [param.to_metadata() for param in params],
            'variants': variants,
        }
    return _render_module(metadata)


def _render_module(metadata: Dict[str, Any]) -> str:
    literal = pprint.pformat(metadata, width=120, sort_dicts=True)
    return f'''# Auto-generated by DaCe. Do not edit.
import base64
import zlib
import cupy
import numpy
_AOT_ABI_VERSION = {AOT_ABI_VERSION}
_KERNELS = {literal}
_MODULE_CACHE = {{}}
_FUNCTION_CACHE = {{}}

def _current_arch():
    return f"sm_{{cupy.cuda.Device().compute_capability}}"

def _validate_metadata(kernel_name, metadata):
    expected_fields = {{"abi_version", "arch", "cubin", "params", "variants"}}
    if set(metadata) != expected_fields:
        raise RuntimeError(
            f"Invalid cuTile AOT metadata fields for {{kernel_name}}: {{set(metadata)!r}}")
    if metadata["abi_version"] != _AOT_ABI_VERSION:
        raise RuntimeError(
            f"Unsupported cuTile AOT ABI version for {{kernel_name}}: "
            f"{{metadata['abi_version']!r}}; expected {{_AOT_ABI_VERSION}}")

    expected_param_fields = {{"kind", "dtype", "ndim", "strides"}}
    for param in metadata["params"]:
        if set(param) != expected_param_fields:
            raise RuntimeError(
                f"Invalid cuTile AOT parameter fields for {{kernel_name}}: {{param!r}}")
        if param["kind"] not in ("array", "scalar") or not isinstance(param["dtype"], str):
            raise RuntimeError(
                f"Invalid cuTile AOT parameter metadata for {{kernel_name}}: {{param!r}}")
        if type(param["ndim"]) is not int or param["ndim"] < 0:
            raise RuntimeError(
                f"Invalid cuTile AOT parameter rank for {{kernel_name}}: {{param!r}}")
        strides = param["strides"]
        if param["kind"] == "scalar":
            if param["ndim"] != 0 or strides is not None:
                raise RuntimeError(
                    f"Invalid cuTile AOT scalar metadata for {{kernel_name}}: {{param!r}}")
        elif param["ndim"] == 0:
            raise RuntimeError(
                f"Invalid cuTile AOT array metadata for {{kernel_name}}: {{param!r}}")
        elif strides is not None and (
                len(strides) != param["ndim"]
                or any(value is not None and (type(value) is not int or value < 0)
                       for value in strides)):
            raise RuntimeError(
                f"Invalid cuTile AOT stride metadata for {{kernel_name}}: {{param!r}}")

    expected_widths = (
        {{"int32", "int64"}}
        if any(param["kind"] == "array" for param in metadata["params"])
        else {{"int32"}})
    actual_widths = {{variant.get("index_dtype") for variant in metadata["variants"]}}
    if actual_widths != expected_widths or any(
            set(variant) != {{"index_dtype", "symbol"}}
            or not isinstance(variant["symbol"], str)
            for variant in metadata["variants"]):
        raise RuntimeError(
            f"Invalid cuTile AOT variants for {{kernel_name}}: {{metadata['variants']!r}}")

def _array_layout(arg, param):
    expected_dtype = numpy.dtype(param["dtype"])
    if not hasattr(arg, "__cuda_array_interface__"):
        raise TypeError(f"cuTile AOT array argument must be on a CUDA device, got {{type(arg).__name__}}")
    if numpy.dtype(arg.dtype) != expected_dtype or arg.ndim != param["ndim"]:
        raise TypeError(f"cuTile AOT array mismatch for {{param}}: dtype={{arg.dtype}}, ndim={{arg.ndim}}")
    if arg.strides is None:
        byte_strides, stride = [], expected_dtype.itemsize
        for size in reversed(arg.shape):
            byte_strides.append(stride)
            stride *= size
        byte_strides.reverse()
    else:
        byte_strides = arg.strides
    if any(stride % expected_dtype.itemsize for stride in byte_strides):
        raise ValueError("cuTile AOT strides must be integral numbers of elements")
    strides = tuple(stride // expected_dtype.itemsize for stride in byte_strides)
    if any(stride < 0 for stride in strides):
        raise ValueError(f"cuTile AOT does not support negative element strides: {{strides}}")
    if param["strides"] is not None and any(
            expected is not None and expected != actual
            for expected, actual in zip(param["strides"], strides)):
        raise ValueError(
            f"cuTile AOT stride mismatch: expected {{param['strides']}}, got {{strides}}")
    return strides

def _fits_int32(args, params):
    low, high = numpy.iinfo(numpy.int32).min, numpy.iinfo(numpy.int32).max
    for arg, param in zip(args, params):
        if param["kind"] == "array":
            if any(v < low or v > high for v in tuple(arg.shape) + _array_layout(arg, param)):
                return False
    return True

def _flatten_args(args, params, index_dtype):
    if len(args) != len(params):
        raise TypeError(f"cuTile AOT expected {{len(params)}} arguments, got {{len(args)}}")
    index_type = numpy.int32 if index_dtype == "int32" else numpy.int64
    flattened = []
    for arg, param in zip(args, params):
        if param["kind"] == "array":
            strides = _array_layout(arg, param)
            flattened.extend((arg, *(index_type(v) for v in arg.shape), *(index_type(v) for v in strides)))
        else:
            flattened.append(numpy.asarray(arg, dtype=numpy.dtype(param["dtype"]))[()])
    return tuple(flattened)

def _cuda_context_key():
    device = cupy.cuda.Device().id
    context = int(cupy.cuda.driver.ctxGetCurrent())
    if context == 0:
        raise RuntimeError("cuTile AOT launch requires a current CUDA context")
    return device, context

def _load_function(kernel_name, metadata, variant):
    context_key = _cuda_context_key()
    module_key = (*context_key, kernel_name)
    module = _MODULE_CACHE.get(module_key)
    if module is None:
        module = cupy.cuda.function.Module()
        module.load(zlib.decompress(base64.b85decode(metadata["cubin"].encode("ascii"))))
        _MODULE_CACHE[module_key] = module
    function_key = (*context_key, kernel_name, variant["symbol"])
    function = _FUNCTION_CACHE.get(function_key)
    if function is None:
        function = module.get_function(variant["symbol"])
        _FUNCTION_CACHE[function_key] = function
    return function

def launch(kernel_name, grid, args):
    metadata = _KERNELS[kernel_name]
    _validate_metadata(kernel_name, metadata)
    current_arch = _current_arch()
    if current_arch != metadata["arch"]:
        raise RuntimeError(f"cuTile AOT cubin for {{kernel_name}} targets {{metadata['arch']}}, current device is {{current_arch}}")
    if len(args) != len(metadata["params"]):
        raise TypeError(f"cuTile AOT expected {{len(metadata['params'])}} arguments, got {{len(args)}}")
    index_dtype = "int32" if _fits_int32(args, metadata["params"]) else "int64"
    variant = next(item for item in metadata["variants"] if item["index_dtype"] == index_dtype)
    flattened = _flatten_args(args, metadata["params"], index_dtype)
    launch_grid = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
    launch_grid += (1,) * (3 - len(launch_grid))
    if len(launch_grid) != 3:
        raise ValueError(f"cuTile AOT grid must have one to three dimensions, got {{launch_grid}}")
    _load_function(kernel_name, metadata, variant)(launch_grid, (1, 1, 1), flattened,
                                                    stream=cupy.cuda.get_current_stream())
'''
