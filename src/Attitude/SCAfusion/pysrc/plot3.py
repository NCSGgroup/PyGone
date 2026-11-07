"""Backward-compatible import path. Prefer pysrc.analysis.plots.calibration."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.plots.calibration", run_name="__main__")
else:
    from pysrc.analysis.plots.calibration import *
