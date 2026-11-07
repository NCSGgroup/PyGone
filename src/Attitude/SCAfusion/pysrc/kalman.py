"""Backward-compatible import path. Prefer pysrc.fusion.kalman."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.fusion.kalman", run_name="__main__")
else:
    from pysrc.fusion.kalman import *
