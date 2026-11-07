"""Backward-compatible import path. Prefer pysrc.fusion.multi_sca."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.fusion.multi_sca", run_name="__main__")
else:
    from pysrc.fusion.multi_sca import *
