"""Backward-compatible import path. Prefer pysrc.data.instruments."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.data.instruments", run_name="__main__")
else:
    from pysrc.data.instruments import *
