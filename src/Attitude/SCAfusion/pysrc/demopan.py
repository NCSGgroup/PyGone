"""Backward-compatible import path. Prefer pysrc.experiments.correlation_demo."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.experiments.correlation_demo", run_name="__main__")
else:
    from pysrc.experiments.correlation_demo import *
