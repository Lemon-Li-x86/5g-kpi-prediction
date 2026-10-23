"""The real-time console survives page/server restarts and detects stale jobs.

These tests do not train models or open a browser.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.ui import web_workflow


class LiveConsoleLifecycleTests(unittest.TestCase):
    def test_pid_detection(self):
        self.assertTrue(web_workflow._worker_alive(os.getpid()))
        for invalid in (0, -1, None, "123", True):
            with self.subTest(invalid=invalid):
                self.assertFalse(web_workflow._worker_alive(invalid))

    def test_finished_child_is_not_live(self):
        process = subprocess.Popen(
            [os.sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        process.wait(timeout=10)
        self.assertFalse(web_workflow._worker_alive(process.pid))

    def test_launch_records_pid_and_detaches_windows_console_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            state = jobs / "current.json"
            session = jobs / "stream_session.json"
            worker = SimpleNamespace(pid=os.getpid())
            with (
                patch.object(web_workflow, "JOBS", jobs),
                patch.object(web_workflow, "STATE", state),
                patch.object(web_workflow, "SESSION", session),
                patch.object(web_workflow, "_busy", return_value=False),
                patch("src.ui.web_workflow.subprocess.Popen", return_value=worker) as popen,
                patch("src.ui.web_workflow.st.success"),
            ):
                web_workflow._launch("download")
            saved = json.loads(session.read_text(encoding="utf-8"))
            self.assertEqual(saved["pid"], os.getpid())
            self.assertEqual(saved["task"], "download")
            self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["status"], "running")
            self.assertEqual(
                popen.call_args.kwargs["creationflags"],
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )


if __name__ == "__main__":
    unittest.main()
