"""Backward-compatible import path. Prefer pysrc.imu.preprocessing."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("pysrc.imu.preprocessing", run_name="__main__")
else:
    from pysrc.imu.preprocessing import *
