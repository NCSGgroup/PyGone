"""Backward-compatible import path. Prefer pysrc.analysis.plots.attitude."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.plots.attitude", run_name="__main__")
else:
    from pysrc.analysis.plots.attitude import *
