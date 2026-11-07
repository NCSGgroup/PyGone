"""Backward-compatible import path. Prefer pysrc.fusion.models.model_three."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.fusion.models.model_three", run_name="__main__")
else:
    from pysrc.fusion.models.model_three import *
