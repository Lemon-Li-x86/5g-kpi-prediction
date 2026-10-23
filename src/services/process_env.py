"""Consistent Unicode and unbuffered output for local Python training jobs.

Windows parent consoles may use GBK. Child training processes must use UTF-8
because the streaming service decodes their stdout/stderr bytes as UTF-8.
"""
from __future__ import annotations

import os


def python_job_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env
