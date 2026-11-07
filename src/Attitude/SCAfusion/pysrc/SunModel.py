"""Backward-compatible import path. Prefer pysrc.analysis.sun_model."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.sun_model", run_name="__main__")
else:
    from pysrc.analysis.sun_model import *
