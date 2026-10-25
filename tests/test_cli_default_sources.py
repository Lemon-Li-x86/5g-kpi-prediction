"""CLI defaults must follow the validated Web defaults without starting training."""
import argparse
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from src.core import training_config


class ParsedDefaults(BaseException):
    pass


class CliDefaultSourceTests(unittest.TestCase):
    def test_cli_consumes_canonical_defaults(self):
        cases = (
            ('train_deep_learning_models', 'DL_DEFAULTS', {'epochs': 17, 'seed': 91}),
            ('detect_kpi_anomalies', 'ANOMALY_DEFAULTS', {'max_events': 1234, 'seed': 91}),
            ('train_traditional_models', 'ML_DEFAULTS', {'max_train': 123456, 'max_test': 999}),
        )
        for module, owner, modified in cases:
            captured = {}

            def capture(parser, *args, **kwargs):
                captured.update({action.dest: action.default for action in parser._actions})
                raise ParsedDefaults

            with self.subTest(module=module), patch.dict(
                getattr(training_config, owner), modified
            ), patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(ParsedDefaults):
                    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/models' /
                                       f'{module}.py'), run_name='__main__')
                for name, expected in modified.items():
                    self.assertEqual(captured[name], expected)


if __name__ == '__main__':
    unittest.main()
