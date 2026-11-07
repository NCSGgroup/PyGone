"""Backward-compatible import path. Prefer pysrc.data.collection."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.data.collection", run_name="__main__")
else:
    from pysrc.data.collection import *
