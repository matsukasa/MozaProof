import unittest

import numpy as np
from PIL import Image, ImageFilter

from src.image_processor import ImageProcessor


class MosaicRenderTests(unittest.TestCase):
    def test_partial_render_preserves_grid_at_right_and_bottom_edges(self):
        source = Image.fromarray(np.random.default_rng(42).integers(
            0, 256, (7, 11, 4), dtype=np.uint8
        ))
        mask = Image.new("L", source.size)
        mask.paste(255, (5, 4, 11, 7))
        fill = Image.new("RGBA", source.size)
        result = ImageProcessor._compose(source, mask, fill, 4, 0)
        self.assertEqual(source.getchannel("A").tobytes(), result.getchannel("A").tobytes())
        pixels = np.asarray(source)
        padded = np.pad(pixels, ((0, 1), (0, 1), (0, 0)), mode="edge")
        for y in range(7):
            for x in range(11):
                if x < 5 or y < 4:
                    self.assertEqual(result.getpixel((x, y)), source.getpixel((x, y)))
                else:
                    row, col = y - y % 4, x - x % 4
                    mean = padded[row:row+4, col:col+4, :3].mean(axis=(0, 1))
                    np.testing.assert_allclose(result.getpixel((x, y))[:3], mean, atol=0.5)

    def test_grayscale_dilation_matches_square_maximum_at_edges(self):
        mask = Image.fromarray(np.random.default_rng(17).integers(
            0, 256, (7, 11), dtype=np.uint8
        ))
        expected = mask.filter(ImageFilter.MaxFilter(5))
        actual = ImageProcessor._dilate_mask_by_pixels(mask, 2)
        self.assertEqual(actual.tobytes(), expected.tobytes())
