"""Backward-compatible CLI and import alias for src.analysis.analyze_anomaly_statistics."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.analysis.analyze_anomaly_statistics", run_name="__main__", alter_sys=True)
else:
    from src.analysis.analyze_anomaly_statistics import *  # noqa: F401,F403
