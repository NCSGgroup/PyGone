"""Backward-compatible import path. Prefer pysrc.experiments.tagged_text_demo."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.experiments.tagged_text_demo", run_name="__main__")
else:
    from pysrc.experiments.tagged_text_demo import *
