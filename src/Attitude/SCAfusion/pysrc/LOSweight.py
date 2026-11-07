"""Backward-compatible import path. Prefer pysrc.fusion.los_weighting."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.fusion.los_weighting", run_name="__main__")
else:
    from pysrc.fusion.los_weighting import *
