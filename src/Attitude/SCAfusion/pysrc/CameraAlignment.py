"""Backward-compatible import path. Prefer pysrc.analysis.camera_alignment."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.analysis.camera_alignment", run_name="__main__")
else:
    from pysrc.analysis.camera_alignment import *
