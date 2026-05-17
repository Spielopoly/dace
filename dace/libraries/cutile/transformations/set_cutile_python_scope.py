"""Mark eligible cuTile map scopes for schedule-based Python cuTile codegen.

This transformation is intentionally conservative. It only marks outer maps
whose scope contains supported cuTile library nodes and structural/data nodes,
and rejects foreign tasklets/library nodes.
"""

from dace import SDFG, dtypes
from dace.sdfg import SDFGState, nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as xf

from dace.libraries.cutile.nodes import (
    TileOpLibraryNode,
    TileRuntimeMaskedOpLibraryNode,
    TileSymbolicMaskedOpLibraryNode,
    TileWhereSelectLibraryNode,
)


class SetCuTilePythonScope(xf.SingleStateTransformation):
    """Mark safe map scopes for cuTile Python backend dispatch."""

    map_entry = xf.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry)]

    @staticmethod
    def _is_supported_cutile_library_node(node: nodes.LibraryNode) -> bool:
        # TODO: make more general by checking for implementations instead of hardcoding node types
        return isinstance(node, (
            TileOpLibraryNode,
            TileRuntimeMaskedOpLibraryNode,
            TileWhereSelectLibraryNode,
        ))

    def can_be_applied(
        self,
        graph: SDFGState,
        expr_index: int,
        sdfg: SDFG,
        permissive: bool = False,
    ) -> bool:
        if getattr(sdfg, "backend", None) != dtypes.BackendLanguage.Python:
            return False

        entry = self.map_entry
        scope = graph.scope_subgraph(entry, include_entry=False, include_exit=False)

        found_cutile_lib = False
        all_already_marked = (entry.map.schedule == dtypes.ScheduleType.CuTile)

        for node in scope.nodes():
            if isinstance(node, (nodes.AccessNode, nodes.MapEntry, nodes.MapExit)):
                continue

            if isinstance(node, nodes.Tasklet):
                # TODO: reconsider
                return False

            if isinstance(node, nodes.NestedSDFG):
                # TODO: CuTile does support function calls (they just get inlined), so we should as well
                return False

            if isinstance(node, nodes.LibraryNode):
                # Correctness-first: symbolic masks are not emitted through
                # this backend path unless supported end-to-end.
                # TODO: refactor
                if isinstance(node, TileSymbolicMaskedOpLibraryNode):
                    return False

                if not self._is_supported_cutile_library_node(node):
                    return False

                if "cutile_python" not in getattr(node, "implementations", {}):
                    return False

                found_cutile_lib = True
                if node.implementation != "cutile_python":
                    all_already_marked = False
                continue

            return False

        return found_cutile_lib and not all_already_marked

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        entry = self.map_entry

        entry.map.schedule = dtypes.ScheduleType.CuTile

        scope = graph.scope_subgraph(entry, include_entry=False, include_exit=False)
        for node in scope.nodes():
            if isinstance(node, nodes.LibraryNode) and self._is_supported_cutile_library_node(node):
                node.implementation = "cutile_python"
