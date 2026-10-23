"""Compatibility entry point for src.services.web_stream_worker."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.services.web_stream_worker", run_name="__main__", alter_sys=True)
else:
    from src.services.web_stream_worker import *  # noqa: F401,F403
