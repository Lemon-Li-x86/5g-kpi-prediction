"""Backward-compatible CLI and import alias for src.analysis.validate_model_robustness."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.analysis.validate_model_robustness", run_name="__main__", alter_sys=True)
else:
    from src.analysis.validate_model_robustness import *  # noqa: F401,F403
