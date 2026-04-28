# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Code I/O stream for Python code generation with whitespace-based indentation.

Unlike the C++ :class:`CodeIOStream` which counts braces to determine
indentation, this stream uses explicit ``indent()`` / ``dedent()`` calls
(or the ``indented()`` context manager) to manage Python's
whitespace-significant blocks.
"""

import io
import inspect
from contextlib import contextmanager

from dace.config import Config
from dace.sdfg.graph import NodeNotFoundError
from dace.sdfg.state import ControlFlowRegion


class PythonCodeIOStream(io.StringIO):
    """Code I/O stream for Python code generation with whitespace-based indentation.

    Indentation is managed explicitly via :meth:`indent`, :meth:`dedent`, and
    the :meth:`indented` context manager.
    
    Example usage:
    >>> stream = PythonCodeIOStream()
    >>> stream.write('def foo():')
    >>> with stream.indented():
    ...     stream.write('print("Hello, world!")')
    >>> print(stream.getvalue())
    def foo():
        print("Hello, world!")
    """

    def __init__(self, base_indentation: int = 0):
        super().__init__()
        self._indent_level: int = base_indentation
        self._spaces = int(Config.get('compiler', 'indentation_spaces'))
        if self._spaces <= 0:
            self._spaces = 4
        self._lineinfo = Config.get_bool('compiler', 'codegen_lineinfo')

    @property
    def indent_level(self) -> int:
        return self._indent_level

    def indent(self, levels: int = 1) -> None:
        """Increase indentation by *levels*."""
        self._indent_level += levels

    def dedent(self, levels: int = 1) -> None:
        """Decrease indentation by *levels* (clamped to 0)."""
        self._indent_level = max(0, self._indent_level - levels)

    @contextmanager
    def indented(self, levels: int = 1):
        """Context manager for scoped indentation."""
        self.indent(levels)
        try:
            yield self
        finally:
            self.dedent(levels)

    def write(self, contents, cfg: 'ControlFlowRegion | None' = None, state_id: int | None = None, node_id: int | None = None) -> int:
        """Write *contents* with proper Python indentation and optional location annotations.

        Each non-empty line is prefixed with the current indentation.  Empty
        lines are emitted as bare newlines.  Relative indentation within
        multi-line *contents* is preserved.  Location annotations use Python
        comment syntax (``# __DACE:…``).
        """
        if contents is None or (isinstance(contents, str) and contents == '' or contents == '\n'):
            return super().write('\n')

        contents = str(contents)

        # Strip a single trailing newline (each line gets its own '\n')
        if contents[-1] == '\n':
            lines = contents[:-1].split('\n')
        else:
            lines = contents.split('\n')

        prefix = ' ' * (self._indent_level * self._spaces)
        

        # Build location annotation (Python comment style)
        location_identifier = ''
        # If SDFG/state/node location is given, annotate this line
        if cfg is not None and self._lineinfo: # TODO: remove and self._lineinfo
            location_identifier = '  #__DACE:%d' % cfg.cfg_id
            if state_id is not None:
                location_identifier += ':' + str(state_id)
                if node_id is not None:
                    if not isinstance(node_id, list):
                        node_id = [node_id]
                    for i, nid in enumerate(node_id):
                        if not isinstance(nid, int):
                            try:
                                state = cfg.state(state_id)
                                node_id[i] = state.node_id(nid)
                            except NodeNotFoundError:
                                node_id[i] = -1
                    location_identifier += ':' + ','.join([str(nid) for nid in node_id])

        if self._lineinfo:
            caller = inspect.getframeinfo(inspect.stack()[1][0], context=0)
            location_identifier += f'  #__CODEGEN {caller.filename}:{caller.lineno}'

        count = 0
        for line in lines:
            if line.strip():
                out_line = prefix + line + location_identifier
                count += super().write(out_line + '\n')
            else:
                count += super().write('\n')

        return count
