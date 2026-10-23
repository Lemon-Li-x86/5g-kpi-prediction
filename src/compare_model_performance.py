"""Backward-compatible CLI and import alias for src.analysis.compare_model_performance."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.analysis.compare_model_performance", run_name="__main__", alter_sys=True)
else:
    from src.analysis.compare_model_performance import *  # noqa: F401,F403
