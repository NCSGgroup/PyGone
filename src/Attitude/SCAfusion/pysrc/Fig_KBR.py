"""Backward-compatible import path. Prefer pysrc.analysis.plots.kbr."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.plots.kbr", run_name="__main__")
else:
    from pysrc.analysis.plots.kbr import *
