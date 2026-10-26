import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.json_io import atomic_write_json
from experiments.run_training_suite import publish_status, read_status


class JsonPublicationTests(unittest.TestCase):
    def test_transient_sharing_error_is_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/"state.json"
            original = Path.replace
            calls = []

            def transient(path, destination):
                calls.append(str(path))
                if len(calls) <= 2:
                    raise PermissionError("file is temporarily open")
                return original(path, destination)

            with patch.object(Path, "replace", new=transient), patch("experiments.json_io.time.sleep"):
                atomic_write_json(target, {"state": "running"})
            self.assertEqual(json.loads(target.read_text()), {"state": "running"})
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_failed_replacement_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/"state.json"
            target.write_text('{"previous": true}')
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), patch("experiments.json_io.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_write_json(target, {"previous": False}, attempts=2)
            self.assertTrue(json.loads(target.read_text())["previous"])

    def test_locked_status_mirror_does_not_fail_training_status(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/"status.json").write_text('{"status": "failed"}')
            original = Path.replace

            def locked_mirror(path, target):
                if Path(target) == suite/"status.json":
                    raise PermissionError("status view held open")
                return original(path, target)

            with patch.object(Path, "replace", new=locked_mirror), patch("experiments.json_io.time.sleep"):
                publish_status(suite, {"status": "running", "completed_runs": ["first"]})
            self.assertEqual(read_status(suite)["status"], "running")
            self.assertEqual(read_status(suite)["completed_runs"], ["first"])


if __name__ == "__main__":
    unittest.main()
