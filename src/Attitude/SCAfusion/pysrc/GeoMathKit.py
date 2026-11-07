"""Backward-compatible import path. Prefer pysrc.core.geometry."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.core.geometry", run_name="__main__")
else:
    from pysrc.core.geometry import *
