"""Backward-compatible import path. Prefer pysrc.core.settings."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.core.settings", run_name="__main__")
else:
    from pysrc.core.settings import *
