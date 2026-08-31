import tempfile
import unittest
from pathlib import Path

import torch

from gesture_model import (
    GestureClassifier,
    GestureDataset,
    GestureModelConfig,
    collate_gesture_batch,
    load_gesture_checkpoint,
    save_gesture_checkpoint,
)


def make_inputs(batch=2, frames=8, features=510, valid_frames=5):
    values = torch.randn(batch, frames, features)
    values[:, valid_frames:] = 0.0
    values[:, :valid_frames, 1] = 1.0
    values[:, :valid_frames, 2] = 0.0
    return values


class GestureModelTests(unittest.TestCase):
    def setUp(self):
        self.config = GestureModelConfig(
            input_size=510,
            num_classes=13,
            lstm1_hidden_size=8,
            lstm2_hidden_size=4,
            attention_heads=2,
            feed_forward_size=8,
            classifier_hidden_size=4,
        )

    def test_output_shape(self):
        model = GestureClassifier(self.config).eval()
        with torch.inference_mode():
            output = model(make_inputs())
        self.assertEqual(output.shape, (2, 13))

    def test_padded_values_do_not_change_output(self):
        model = GestureClassifier(self.config).eval()
        first = make_inputs(batch=1)
        second = first.clone()
        second[:, 5:, 3:] = torch.randn_like(second[:, 5:, 3:]) * 100

        with torch.inference_mode():
            first_output = model(first)
            second_output = model(second)
        self.assertTrue(torch.allclose(first_output, second_output))

    def test_checkpoint_round_trip(self):
        model = GestureClassifier(self.config).eval()
        inputs = make_inputs(batch=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_gesture_checkpoint(path, model, {"version": 10})
            restored, metadata = load_gesture_checkpoint(path, "cpu")

            with torch.inference_mode():
                expected = model(inputs)
                actual = restored(inputs)

        self.assertEqual(metadata["version"], 10)
        self.assertTrue(torch.allclose(expected, actual))

    def test_collate_pads_only_to_longest_gesture(self):
        short = make_inputs(batch=1, frames=3, valid_frames=3)[0]
        long = make_inputs(batch=1, frames=7, valid_frames=7)[0]
        dataset = GestureDataset([short, long], [0, 1])

        inputs, labels = collate_gesture_batch([dataset[0], dataset[1]])

        self.assertEqual(inputs.shape, (2, 7, 510))
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertTrue(torch.all(inputs[0, 3:] == 0))


if __name__ == "__main__":
    unittest.main()
