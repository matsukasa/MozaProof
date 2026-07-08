from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.auto_censor import AutoCensorEngine, HotscreenDetector


class FakeDetector:
    def detect(self, _path: str):
        return [
            {
                "class": "FEMALE_GENITALIA_EXPOSED",
                "score": 0.9,
                "box": [20, 20, 30, 20],
            }
        ]


class FakeSegmenter:
    def __init__(self, _model_dir: Path) -> None:
        pass

    def encode(self, image_bgr):
        return {"shape": image_bgr.shape[:2]}

    def segment_box(self, encoded, box):
        height, width = encoded["shape"]
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        return np.asarray(mask) > 0


class AutoCensorEngineTests(unittest.TestCase):
    def test_detected_box_has_no_automatic_margin(self) -> None:
        self.assertEqual(
            AutoCensorEngine._detected_box([20, 20, 30, 20], (100, 80)),
            (20, 20, 50, 40),
        )

    def test_hotscreen_preprocess_and_coordinate_restore(self) -> None:
        image = np.zeros((320, 640, 3), dtype=np.uint8)
        tensor, scale, pad = HotscreenDetector._preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(scale, 1.0)
        self.assertEqual(pad, (0, 160))

        output = np.zeros((1, 20, 2), dtype=np.float32)
        output[0, 0:4, 0] = (320, 320, 100, 80)
        output[0, 7, 0] = 0.9  # class 3 + four box values
        detections = HotscreenDetector._postprocess(
            output, image.shape[:2], scale, pad, confidence=0.15
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class"], "FEMALE_GENITALIA_EXPOSED")
        self.assertEqual(detections[0]["box"], [270, 120, 100, 80])

    def test_real_hotscreen_model_loads_and_runs(self) -> None:
        detector = HotscreenDetector(Path("models"))
        detections = detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertIsInstance(detections, list)

    def test_hotscreen_deduplicates_overlapping_tiles(self) -> None:
        detections = [
            {
                "class": "MALE_GENITALIA_EXPOSED",
                "score": 0.9,
                "box": [100, 100, 80, 100],
            },
            {
                "class": "MALE_GENITALIA_EXPOSED",
                "score": 0.7,
                "box": [104, 104, 80, 100],
            },
            {
                "class": "FEMALE_GENITALIA_EXPOSED",
                "score": 0.8,
                "box": [300, 200, 60, 70],
            },
        ]
        merged = HotscreenDetector._deduplicate(detections, confidence=0.15)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["score"], 0.9)

    def test_detection_becomes_non_rectangular_shape_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), "white").save(path)
            engine = AutoCensorEngine(
                detector=FakeDetector(), segmenter_factory=FakeSegmenter
            )
            result = engine.detect(path)
        self.assertEqual(result.detection_count, 1)
        self.assertEqual(result.segmented_count, 1)
        bbox = result.mask.getbbox()
        self.assertIsNotNone(bbox)
        self.assertEqual(result.mask.getpixel((bbox[0], bbox[1])), 0)
        self.assertEqual(
            result.mask.getpixel(((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)),
            255,
        )

    def test_unrelated_detection_is_ignored_without_loading_segmenter(self) -> None:
        class SafeDetector:
            def detect(self, _path):
                return [{"class": "FACE_FEMALE", "score": 0.99, "box": [1, 1, 5, 5]}]

        def fail_factory(_model_dir):
            raise AssertionError("segmenter should not load")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (20, 20), "white").save(path)
            result = AutoCensorEngine(
                detector=SafeDetector(), segmenter_factory=fail_factory
            ).detect(path)
        self.assertEqual(result.detection_count, 0)
        self.assertIsNone(result.mask.getbbox())

    def test_anus_detection_is_not_censored_automatically(self) -> None:
        class AnusDetector:
            def detect(self, _image):
                return [
                    {
                        "class": "ANUS_EXPOSED",
                        "score": 0.99,
                        "box": [1, 1, 10, 10],
                    }
                ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (20, 20), "white").save(path)
            result = AutoCensorEngine(detector=AnusDetector()).detect(path)
        self.assertEqual(result.detection_count, 0)
        self.assertIsNone(result.mask.getbbox())


if __name__ == "__main__":
    unittest.main()
