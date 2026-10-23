"""Backward-compatible CLI and import alias for src.analysis.review_model_results."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.analysis.review_model_results", run_name="__main__", alter_sys=True)
else:
    from src.analysis.review_model_results import *  # noqa: F401,F403
