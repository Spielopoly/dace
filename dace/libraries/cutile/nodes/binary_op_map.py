"""
Masked cuTile binary operation library nodes.

Implements element-wise masked binary operations:
	if mask[idx]:
		C[idx] = OP(A[idx], B[idx])

When the mask is false, the output element is left untouched.
"""
from __future__ import annotations

from typing import Callable, cast

import dace
from dace import dtypes, properties
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation


@library.node
class TileMaskedBinaryOPLibraryNode(LibraryNode):
	"""
	Library node for masked binary operation on 2 tiles:
		if M: C = foo(A, B)

	Connectors
	----------
	_a  (in)  : tile A
	_b  (in)  : tile B
	_m  (in)  : tile mask (same shape as A/B/C)
	_c  (out) : tile C
	"""

	implementations: dict = {}
	default_implementation = "pure"

	tile_shape = properties.ListProperty(
		element_type=int,
		default=None,
		desc=(
			"Tile dimensions, e.g. [128] for a 1-D tile or [32, 32] for 2-D."
			"0-D (scalar) tiles can be represented with an empty list []."
			"When None, the shape is inferred from the incoming array descriptors "
			"at expansion time."
		),
		allow_none=True,
	)

	def __init__(self,
				 name: str = "TileMaskedBinaryOP",
				 tile_shape: list[int] | None = None,
				 **kwargs):
		super().__init__(
			name,
			inputs={"_a", "_b", "_m"},
			outputs={"_c"},
			**kwargs,
		)
		self.tile_shape = tile_shape

	def validate(self, sdfg: SDFG, state: SDFGState):
		a_node = b_node = m_node = c_node = None
		for edge in state.in_edges(self):
			if edge.dst_conn == "_a":
				a_node = edge.src
			elif edge.dst_conn == "_b":
				b_node = edge.src
			elif edge.dst_conn == "_m":
				m_node = edge.src
		for edge in state.out_edges(self):
			if edge.src_conn == "_c":
				c_node = edge.dst

		if a_node is None or b_node is None or m_node is None or c_node is None:
			raise InvalidSDFGNodeError(
				f"TileMaskedBinaryOP '{self.name}': connectors (_a, _b, _m, _c) must all be connected.",
				sdfg=sdfg,
				state_id=state.parent_graph.node_id(state),
				node_id=state.node_id(self),
			)

		a_desc = sdfg.arrays[a_node.data]
		b_desc = sdfg.arrays[b_node.data]
		m_desc = sdfg.arrays[m_node.data]
		c_desc = sdfg.arrays[c_node.data]

		if a_desc.shape != b_desc.shape or a_desc.shape != c_desc.shape or a_desc.shape != m_desc.shape:
			raise InvalidSDFGNodeError(
				f"TileMaskedBinaryOP '{self.name}': shape mismatch - "
				f"A={a_desc.shape}, B={b_desc.shape}, M={m_desc.shape}, C={c_desc.shape}",
				sdfg=sdfg,
				state_id=state.parent_graph.node_id(state),
				node_id=state.node_id(self),
			)

		if a_desc.dtype != b_desc.dtype:
			raise InvalidSDFGNodeError(
				f"TileMaskedBinaryOP '{self.name}': dtype mismatch - A={a_desc.dtype}, B={b_desc.dtype}",
				sdfg=sdfg,
				state_id=state.parent_graph.node_id(state),
				node_id=state.node_id(self),
			)
		if c_desc.dtype != a_desc.dtype:
			raise InvalidSDFGNodeError(
				f"TileMaskedBinaryOP '{self.name}': dtype mismatch - A={a_desc.dtype}, C={c_desc.dtype}",
				sdfg=sdfg,
				state_id=state.parent_graph.node_id(state),
				node_id=state.node_id(self),
			)

		supported_mask_dtypes = {
			dace.bool,
			dace.int8,
			dace.uint8,
			dace.int16,
			dace.uint16,
			dace.int32,
			dace.uint32,
			dace.int64,
			dace.uint64,
		}
		if m_desc.dtype not in supported_mask_dtypes:
			raise InvalidSDFGNodeError(
				f"TileMaskedBinaryOP '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
				sdfg=sdfg,
				state_id=state.parent_graph.node_id(state),
				node_id=state.node_id(self),
			)


class ExpandTileElementWiseMaskedBinaryOPPure(ExpandTransformation):
	"""Expand TileMaskedBinaryOPLibraryNode into a C++ tasklet."""

	environments: list = []

	@staticmethod
	def expansion_with_op(node: LibraryNode, parent_state: SDFGState,
						  parent_sdfg: SDFG,
						  binary_op_string_generator: Callable[[str, str], str]):
		masked_node = cast(TileMaskedBinaryOPLibraryNode, node)
		a_desc, b_desc, m_desc, c_desc = _get_masked_tile_descriptors(masked_node, parent_state, parent_sdfg)

		tile_shape = getattr(masked_node, "tile_shape", None)
		shape: tuple[int] = tuple(tile_shape) if tile_shape is not None else a_desc.shape
		ndim: int = len(shape)

		try:
			n_total = 1
			for s in shape:
				n_total *= int(s)
			use_scalar_form = (ndim == 0) or (n_total == 1)
		except (TypeError, ValueError):
			use_scalar_form = (ndim == 0)

		if use_scalar_form:
			code = f"if (_m) {{ _c = {binary_op_string_generator('_a', '_b')}; }}"
		else:
			shape_expr = ", ".join(symstr(s) for s in shape)
			a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
			b_strides_expr = ", ".join(symstr(s) for s in b_desc.strides)
			m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
			c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

			code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t b_strides[ndim] = {{{b_strides_expr}}};
const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
	n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
	std::size_t rem = i;
	std::size_t ia = 0;
	std::size_t ib = 0;
	std::size_t im = 0;
	std::size_t ic = 0;
	for (int d = ndim - 1; d >= 0; --d) {{
		const auto extent = shape[d];
		const std::size_t coord = rem % extent;
		rem /= extent;
		ia += coord * a_strides[d];
		ib += coord * b_strides[d];
		im += coord * m_strides[d];
		ic += coord * c_strides[d];
	}}
	if (_m[im]) {{
		_c[ic] = {binary_op_string_generator('_a[ia]', '_b[ib]')};
	}}
}}
"""

		tasklet = nodes.Tasklet(
			label=masked_node.name + "_cutile",
			inputs={"_a", "_b", "_m"},
			outputs={"_c"},
			code=code,
			language=dtypes.Language.CPP,
		)
		return tasklet


def _get_masked_tile_descriptors(node: TileMaskedBinaryOPLibraryNode, state: SDFGState,
								 sdfg: SDFG) -> tuple[dace.data.Data, dace.data.Data, dace.data.Data, dace.data.Data]:
	"""Return (a_desc, b_desc, m_desc, c_desc) array descriptors for node."""
	a_desc = b_desc = m_desc = c_desc = None
	for edge in state.in_edges(node):
		arr_name = edge.data.data
		if arr_name is None:
			continue
		if edge.dst_conn == "_a":
			a_desc = sdfg.arrays[arr_name]
		elif edge.dst_conn == "_b":
			b_desc = sdfg.arrays[arr_name]
		elif edge.dst_conn == "_m":
			m_desc = sdfg.arrays[arr_name]
	for edge in state.out_edges(node):
		arr_name = edge.data.data
		if arr_name is None:
			continue
		if edge.src_conn == "_c":
			c_desc = sdfg.arrays[arr_name]
	if None in (a_desc, b_desc, m_desc, c_desc):
		raise ValueError(
			f"TileMaskedAdd expansion: could not resolve all array descriptors for "
			f"node '{node.name}'. Make sure _a, _b, _m, _c are all connected."
		)
	return (
		cast(dace.data.Data, a_desc),
		cast(dace.data.Data, b_desc),
		cast(dace.data.Data, m_desc),
		cast(dace.data.Data, c_desc),
	)


@library.node
class TileMaskedAddLibraryNode(TileMaskedBinaryOPLibraryNode):
	"""Masked tile addition: if M then C = A + B else keep C unchanged."""

	def __init__(self,
				 name: str = "TileMaskedAdd",
				 tile_shape: list[int] | None = None,
				 **kwargs):
		super().__init__(
			name=name,
			tile_shape=tile_shape,
			**kwargs,
		)


@library.register_expansion(TileMaskedAddLibraryNode, "pure")  # type: ignore[arg-type]
class ExpandTileMaskedAddPure(ExpandTileElementWiseMaskedBinaryOPPure):
	"""Expand TileMaskedAddLibraryNode into a C++ tasklet."""

	@staticmethod
	def expansion(node: TileMaskedAddLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
		return ExpandTileElementWiseMaskedBinaryOPPure.expansion_with_op(
			node,
			state,
			sdfg,
			binary_op_string_generator=lambda a, b: f"{a} + {b}",
		)
