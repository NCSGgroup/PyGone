"""Backward-compatible import path. Prefer pysrc.experiments.text2_demo."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.experiments.text2_demo", run_name="__main__")
else:
    from pysrc.experiments.text2_demo import *
