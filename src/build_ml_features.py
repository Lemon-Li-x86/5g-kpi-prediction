"""Backward-compatible CLI and import alias for src.data.build_ml_features."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.data.build_ml_features", run_name="__main__", alter_sys=True)
else:
    from src.data.build_ml_features import *  # noqa: F401,F403
