"""Backward-compatible import path. Prefer pysrc.analysis.hfn_correlation."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.hfn_correlation", run_name="__main__")
else:
    from pysrc.analysis.hfn_correlation import *
