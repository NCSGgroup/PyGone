"""Backward-compatible import path. Prefer pysrc.fusion.iteration."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.fusion.iteration", run_name="__main__")
else:
    from pysrc.fusion.iteration import *
