"""Backward-compatible import path. Prefer pysrc.analysis.plots.legacy_summary."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.plots.legacy_summary", run_name="__main__")
else:
    from pysrc.analysis.plots.legacy_summary import *
