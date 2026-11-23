import unittest
from unittest.mock import Mock

import numpy as np

from src.auto_censor import HotscreenDetector


class DetectionConfirmationTests(unittest.TestCase):
    def test_only_a_matching_confident_crop_can_recover_a_weak_target(self):
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        target = "MALE_GENITALIA_EXPOSED"
        candidate = {"class": target, "box": [200, 200, 40, 40], "score": 0.18}
        for label, box, score, accepted in [
            (target, [40, 40, 40, 40], 0.52, True),
            (target, [40, 40, 40, 40], 0.20, False),
            (target, [0, 0, 20, 20], 0.90, False),
            ("FEMALE_FACE", [40, 40, 40, 40], 0.90, False),
        ]:
            with self.subTest(label=label, box=box, score=score):
                detector = object.__new__(HotscreenDetector)
                detector._detect_single = Mock(side_effect=[
                    [candidate], [{"class": label, "box": box, "score": score}]
                ])
                results = detector.detect(image)
                self.assertEqual(len(results), int(accepted))
                self.assertEqual(detector._detect_single.call_count, 2)
                if accepted:
                    self.assertEqual(results[0]["score"], score)
                    self.assertEqual(results[0]["box"], candidate["box"])
