import json
import mmap
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.checkpoints import resolve_best_checkpoint, save_best_checkpoint
from experiments.run_training_suite import next_available_directory, training_directory
from gesture_model import GestureClassifier, GestureModelConfig


class CheckpointRecoveryTests(unittest.TestCase):
    def test_new_checkpoints_save_while_old_best_is_mapped(self):
        model = GestureClassifier(GestureModelConfig(input_size=3, num_classes=2))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            old = directory/"best.pt"
            old.write_bytes(b"previous checkpoint is held open")
            with old.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ):
                first = save_best_checkpoint(directory, 1, model, {"epoch": 1})
                second = save_best_checkpoint(directory, 2, model, {"epoch": 2})
                self.assertEqual(resolve_best_checkpoint(directory), second.resolve())
                self.assertTrue(first.exists())
                self.assertEqual(old.read_bytes(), b"previous checkpoint is held open")
                self.assertEqual(torch.load(second, weights_only=True)["metadata"]["epoch"], 2)
                with self.assertRaises(FileExistsError):
                    save_best_checkpoint(directory, 2, model, {"epoch": 2})

    def test_failed_save_does_not_advance_best_pointer(self):
        model = GestureClassifier(GestureModelConfig(input_size=3, num_classes=2))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = save_best_checkpoint(directory, 1, model, {"epoch": 1})
            with patch("experiments.checkpoints.save_gesture_checkpoint", side_effect=RuntimeError("disk unavailable")):
                with self.assertRaises(RuntimeError):
                    save_best_checkpoint(directory, 2, model, {"epoch": 2})
            self.assertEqual(resolve_best_checkpoint(directory), first.resolve())

    def test_pointer_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/"best_checkpoint.json").write_text(json.dumps({"checkpoint": "../other.pt"}))
            with self.assertRaises(ValueError):
                resolve_best_checkpoint(directory)

    def test_retry_preserves_failed_attempt_and_reuses_completed_training(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/"run"
            base.mkdir()
            retry, reuse = training_directory(base)
            self.assertEqual(retry, base.with_name("run_retry1"))
            self.assertFalse(reuse)
            retry.mkdir()
            (retry/"training_complete.json").write_text("{}")
            self.assertEqual(training_directory(base), (retry, True))
            self.assertEqual(next_available_directory(base), base.with_name("run_retry2"))


if __name__ == "__main__":
    unittest.main()
