import unittest

import numpy as np

from gesture_preprocessing import (
    HAND_FEATURES,
    GestureFeatureBuilder,
    build_model_input,
)


def make_hand(offset=0.0):
    return [
        {"x": offset + i / 100.0, "y": i / 200.0, "z": -i / 300.0}
        for i in range(21)
    ]


class GesturePreprocessingTests(unittest.TestCase):
    def test_first_frame_has_zero_motion(self):
        builder = GestureFeatureBuilder()
        row = builder.build_frame(
            [({"categoryName": "Right"}, make_hand())],
            0,
            timestamp_ms=1000,
        )

        speed_values = []
        for landmark_index in range(21):
            start = 3 + (landmark_index * 6) + 3
            speed_values.extend(row[start : start + 3])

        self.assertTrue(np.allclose(speed_values, 0.0))

    def test_hand_slots_do_not_depend_on_detector_order(self):
        right = make_hand(0.1)
        left = make_hand(0.7)

        first = GestureFeatureBuilder().build_frame(
            [({"categoryName": "Left"}, left), ({"categoryName": "Right"}, right)],
            0,
            timestamp_ms=1000,
        )
        second = GestureFeatureBuilder().build_frame(
            [({"categoryName": "Right"}, right), ({"categoryName": "Left"}, left)],
            0,
            timestamp_ms=1000,
        )

        self.assertEqual(first, second)
        self.assertEqual(first[1:3], [1, 1])

    def test_model_input_keeps_exact_detected_length(self):
        builder = GestureFeatureBuilder()
        frames = [
            builder.build_frame(
                [({"categoryName": "Right"}, make_hand(i / 10))],
                i,
                timestamp_ms=1000 + (i * 100),
            )
            for i in range(2)
        ]

        raw, model_input = build_model_input(frames)

        self.assertEqual(len(frames), 2)
        self.assertEqual(raw.shape, (2, 255))
        self.assertEqual(model_input.shape, (1, 2, 510))
        self.assertTrue(np.allclose(raw, np.asarray(frames, dtype=np.float32)))

    def test_missing_hand_has_zero_filled_slot(self):
        row = GestureFeatureBuilder().build_frame(
            [({"categoryName": "Left"}, make_hand())],
            0,
            timestamp_ms=1000,
        )

        self.assertEqual(row[1:3], [0, 1])
        self.assertTrue(np.allclose(row[3 : 3 + HAND_FEATURES], 0.0))

    def test_motion_is_normalized_by_elapsed_time(self):
        builder = GestureFeatureBuilder()
        first = make_hand()
        second = make_hand()
        second[0]["x"] += 0.1

        builder.build_frame(
            [({"categoryName": "Right"}, first)],
            0,
            timestamp_ms=1000,
        )
        row = builder.build_frame(
            [({"categoryName": "Right"}, second)],
            1,
            timestamp_ms=1200,
        )

        wrist_speed_x = row[6]
        self.assertAlmostEqual(wrist_speed_x, 0.5)


if __name__ == "__main__":
    unittest.main()
