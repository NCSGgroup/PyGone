"""Backward-compatible import path. Prefer pysrc.analysis.spectral."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.spectral", run_name="__main__")
else:
    from pysrc.analysis.spectral import *
