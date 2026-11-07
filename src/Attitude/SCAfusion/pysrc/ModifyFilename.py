"""Backward-compatible import path. Prefer pysrc.tools.modify_filename."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.tools.modify_filename", run_name="__main__")
else:
    from pysrc.tools.modify_filename import *
