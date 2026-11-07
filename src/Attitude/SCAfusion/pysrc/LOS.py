"""Backward-compatible import path. Prefer pysrc.attitude.los."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.attitude.los", run_name="__main__")
else:
    from pysrc.attitude.los import *
