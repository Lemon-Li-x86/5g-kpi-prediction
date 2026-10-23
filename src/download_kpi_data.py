"""Backward-compatible CLI and import alias for src.data.download_kpi_data."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.data.download_kpi_data", run_name="__main__", alter_sys=True)
else:
    from src.data.download_kpi_data import *  # noqa: F401,F403
