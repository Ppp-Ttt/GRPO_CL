"""Minimal compatibility shim for `pyext` on modern Python versions.

The upstream `pyext` package is unmaintained and unavailable on modern
Python versions in this environment. PRIME code evaluation only relies on
`RuntimeModule.from_string`, so we provide that subset here.
"""

from __future__ import annotations

import sys
import types


class RuntimeModule(types.ModuleType):
    @classmethod
    def from_string(cls, module_name: str, file_path: str, source: str) -> "RuntimeModule":
        module = cls(module_name)
        filename = file_path or "<string>"
        module.__file__ = filename
        code = compile(source, filename, "exec")
        sys.modules[module_name] = module
        exec(code, module.__dict__)
        return module