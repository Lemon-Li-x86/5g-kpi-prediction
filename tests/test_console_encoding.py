"""Exercise the same UTF-8 settings used for browser-launched training."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.services.process_env import python_job_env


class ConsoleEncodingTests(unittest.TestCase):
    def test_windows_gbk_parent_is_overridden(self):
        with patch.dict(os.environ, {"PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0"}):
            env = python_job_env()
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    def test_child_prints_unicode_losslessly(self):
        env = python_job_env()
        done = subprocess.run(
            [sys.executable, "-c", "print('R²: 0.654321 | 训练完成')"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        self.assertIn("R²: 0.654321", done.stdout.decode("utf-8"))
        self.assertIn("训练完成", done.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
