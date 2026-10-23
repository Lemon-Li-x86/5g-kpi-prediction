"""Backward-compatible CLI and import alias for src.models.train_deep_learning_models."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.models.train_deep_learning_models", run_name="__main__", alter_sys=True)
else:
    from src.models.train_deep_learning_models import *  # noqa: F401,F403
