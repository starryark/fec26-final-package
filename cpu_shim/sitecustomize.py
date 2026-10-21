"""CPU portability shim (loaded automatically when this directory is on PYTHONPATH).

Active only with FEC_DEVICE=cpu (infer.py --device cpu).  The production code defaults its torch
solvers and CLIs to device "cuda"; this shim rewrites those DEFAULTS to "cpu" so the same code runs
on a machine without a GPU:
  * every function / method default argument equal to "cuda" in an imported fec.* module;
  * argparse arguments declared with default="cuda".
Explicit device arguments are untouched (fec/run_final.py passes "cpu" itself under FEC_DEVICE).
CPU results equal the CUDA results only to float32 rounding (see PROVENANCE.md, determinism).
"""
import os

if os.environ.get("FEC_DEVICE", "") == "cpu":
    import argparse
    import importlib.abc
    import sys
    import types

    _orig_add = argparse._ActionsContainer.add_argument

    def _add_argument(self, *args, **kw):
        if kw.get("default") == "cuda":
            kw["default"] = "cpu"
        return _orig_add(self, *args, **kw)

    argparse._ActionsContainer.add_argument = _add_argument

    def _fix_func(f):
        d = getattr(f, "__defaults__", None)
        if d and "cuda" in [x for x in d if isinstance(x, str)]:
            f.__defaults__ = tuple("cpu" if (isinstance(x, str) and x == "cuda") else x for x in d)
        kd = getattr(f, "__kwdefaults__", None)
        if kd:
            for k, v in list(kd.items()):
                if isinstance(v, str) and v == "cuda":
                    kd[k] = "cpu"

    def _patch_module(mod):
        for v in list(vars(mod).values()):
            if isinstance(v, types.FunctionType) and v.__module__ == mod.__name__:
                _fix_func(v)
            elif isinstance(v, type) and v.__module__ == mod.__name__:
                for a in vars(v).values():
                    if isinstance(a, (staticmethod, classmethod)):
                        a = a.__func__
                    if isinstance(a, types.FunctionType):
                        _fix_func(a)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("fec."):
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            if loader is None or not hasattr(loader, "exec_module"):
                return spec
            orig = loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                _patch_module(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
