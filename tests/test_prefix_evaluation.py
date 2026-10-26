import unittest

import numpy as np

from experiments.evaluate_prefixes import class_mapping, prefix_input, validate_raw
from experiments.features import feature_view


class PrefixEvaluationTests(unittest.TestCase):
    def raw(self):
        values = np.zeros((10, 255), dtype=np.float32)
        values[:, 0] = np.arange(10)
        values[:, 1] = 1
        values[:, 3] = np.arange(10)
        return values

    def test_prefix_does_not_use_future_frames(self):
        raw = self.raw()
        first = prefix_input(raw, fraction=0.3)
        raw[3:, 3:] = 999
        second = prefix_input(raw, fraction=0.3)
        np.testing.assert_array_equal(first.numpy(), second.numpy())
        self.assertEqual(tuple(first.shape), (3, 510))
        np.testing.assert_array_equal(first.numpy()[0, 255:], 0)

    def test_round_up_and_cap_short_clips(self):
        self.assertEqual(len(prefix_input(self.raw(), fraction=0.11)), 2)
        self.assertEqual(len(prefix_input(self.raw(), frames=31)), 10)

    def test_rejects_incompatible_class_maps(self):
        embedded = {"parameters": {"numbered_unique_words": {"A": 0, "B": 1}}}
        with self.assertRaises(ValueError):
            class_mapping(embedded, {"numbered_unique_words": {"A": 1, "B": 0}}, 2)
        with self.assertRaises(ValueError):
            class_mapping(embedded, None, 3)

    def test_rejects_internal_all_absent_frames(self):
        raw = self.raw()
        raw[3, 1:3] = 0
        with self.assertRaises(ValueError):
            validate_raw(raw)

    def test_feature_variants_keep_presence_and_remove_frame_index(self):
        raw = self.raw()
        for variant, width in (("positions", 129), ("scaled-positions", 129),
                               ("positions-diff", 255), ("positions-motion", 255)):
            features = feature_view(raw, variant)
            self.assertEqual(features.shape, (10, width))
            np.testing.assert_array_equal(features[:, 0], 0)
            np.testing.assert_array_equal(features[:, 1:3], raw[:, 1:3])

    def test_scale_normalization_is_invariant_to_uniform_hand_scale(self):
        raw = self.raw()
        raw[:, 3+9*6] = 2
        scaled = raw.copy()
        scaled[:, 3:] *= 3
        np.testing.assert_allclose(feature_view(raw, "scaled-positions"),
                                   feature_view(scaled, "scaled-positions"))

    def test_motion_differences_reset_after_absent_hand(self):
        raw = self.raw()
        raw[:, 2] = 1
        raw[3, 1] = 0
        features = feature_view(raw, "positions-diff")
        np.testing.assert_array_equal(features[4, 129:192], 0)


if __name__ == "__main__":
    unittest.main()
