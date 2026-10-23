"""Backward-compatible CLI and import alias for src.models.detect_kpi_anomalies."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.models.detect_kpi_anomalies", run_name="__main__", alter_sys=True)
else:
    from src.models.detect_kpi_anomalies import *  # noqa: F401,F403
