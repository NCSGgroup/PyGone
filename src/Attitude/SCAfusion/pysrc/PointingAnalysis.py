"""Backward-compatible import path. Prefer pysrc.attitude.pointing."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.attitude.pointing", run_name="__main__")
else:
    from pysrc.attitude.pointing import *
