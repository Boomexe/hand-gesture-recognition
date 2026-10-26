"""Website presentation/API regression checks; no training or real predictions."""
import importlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class WebsiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("website.app")
        cls.module.app.config["TESTING"] = True

    def setUp(self):
        self.client = self.module.app.test_client()

    def test_workspace_pages_and_dictionary(self):
        for route, page in [("/", "live"), ("/by-video", "video"),
                            ("/by-video-continuous", "video"), ("/library", "library")]:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)
                html = response.get_data(as_text=True)
                self.assertIn('data-page="' + page + '"', html)
                self.assertIn("Signbridge", html)
                match = re.search(r'<script id="app-config" type="application/json">(.*?)</script>', html)
                config = json.loads(match.group(1))
                self.assertEqual(config["words"], sorted(set(self.module.words.values())))
                self.assertEqual(config["confidenceThreshold"], self.module.THRESHOLD)
                self.assertNotIn(str(self.module.model_path), html)

    def test_video_mode_defaults(self):
        for route, checked in [("/by-video", False), ("/by-video-continuous", True)]:
            html = self.client.get(route).get_data(as_text=True)
            element = re.search(r'<input id="prefix-mode"[^>]+>', html).group()
            self.assertEqual("checked" in element, checked)

    def test_pages_use_functional_titles_without_promotional_copy(self):
        for route, title in [("/", "Live studio"), ("/by-video", "Video workspace"),
                             ("/library", "Gesture library")]:
            with self.subTest(route=route):
                html = self.client.get(route).get_data(as_text=True)
                self.assertIn("<h1>" + title + "</h1>", html)
                for removed in ["Make a sign.", "Find the <em>word.", "A closer look",
                                "Small gestures.", "Real <em>meaning.", "Ready when you are.",
                                'class="section-kicker"', 'class="intro-description"',
                                'class="sidebar-note"']:
                    self.assertNotIn(removed, html)

    def test_missing_file_is_json(self):
        for route in ["/upload-video", "/upload-video-continuous"]:
            response = self.client.post(route)
            self.assertEqual(response.status_code, 400)
            self.assertIn("message", response.json)

    def test_missing_model_is_reported(self):
        with patch.object(self.module, "model", None):
            response = self.client.post("/upload-video", data={"video": (io.BytesIO(b"clip"), "clip.mp4")})
        self.assertEqual(response.status_code, 503)

    def test_prefix_results_include_full_gesture_and_cleanup_upload(self):
        frames = np.zeros((1, 3, self.module.MODEL_DATA_PER_FRAME), dtype=np.float32)
        prediction = {"predicted_word": "MORE", "confidence": 0.95}
        lengths = []
        paths = []
        def preprocess(path):
            paths.append(path)
            self.assertTrue(path.exists())
            return frames
        def predict(values):
            lengths.append(values.shape[1])
            return prediction
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.module, "WEBSITE_ROOT", Path(directory)), \
                 patch.object(self.module, "model", object()), \
                 patch.object(self.module, "preprocess_video_from_upload", side_effect=preprocess), \
                 patch.object(self.module, "prediction_result", side_effect=predict):
                response = self.client.post("/upload-video-continuous", data={"video": (io.BytesIO(b"clip"), "clip.mov")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(lengths, [1, 2, 3])
            self.assertEqual([item["frame"] for item in response.json], [0, 1, 2])
            self.assertEqual(response.json[-1]["pred_word"], "MORE")
            self.assertFalse(paths[0].exists())
            self.assertEqual(list((Path(directory) / "uploads").iterdir()), [])

    def test_full_video_result_and_error_cleanup(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                frames = np.zeros((1, 2, self.module.MODEL_DATA_PER_FRAME), dtype=np.float32)
                with patch.object(self.module, "WEBSITE_ROOT", Path(directory)), \
                     patch.object(self.module, "model", object()), \
                     patch.object(self.module, "preprocess_video_from_upload", **({"side_effect": ValueError("No hands detected in video.")} if fail else {"return_value": frames})), \
                     patch.object(self.module, "prediction_result", return_value={"predicted_word": "MORE", "confidence": .95}) as predict, \
                     patch.object(self.module.app.logger, "exception"):
                    response = self.client.post("/upload-video", data={"video": (io.BytesIO(b"clip"), "clip.mp4")})
                self.assertEqual(response.status_code, 500 if fail else 200)
                self.assertTrue(response.is_json)
                self.assertEqual(predict.call_count, 0 if fail else 1)
                self.assertEqual(list((Path(directory) / "uploads").iterdir()), [])

    def test_oversized_upload_is_json(self):
        with patch.dict(self.module.app.config, {"MAX_CONTENT_LENGTH": 32}):
            response = self.client.post("/upload-video", data={"video": (io.BytesIO(b"x" * 128), "clip.mp4")})
        self.assertEqual(response.status_code, 413)
        self.assertIn("too large", response.json["message"])

    def test_socket_reset_still_works(self):
        client = self.module.socketio.test_client(self.module.app)
        try:
            client.emit("reset_live")
            self.assertTrue(any(item["name"] == "prediction" and item["args"][0]["status"] == "reset" for item in client.get_received()))
        finally:
            client.disconnect()


if __name__ == "__main__":
    unittest.main()
