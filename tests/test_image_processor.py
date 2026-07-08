from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from src.history import HistoryStack
from src.image_processor import ImageLoadError, ImageProcessor, OriginalOverwriteError
from src.settings import pixiv_block_size, pixiv_brush_size, scaled_mosaic_block_size
from src.tools import ToolType


class ImageProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_png(self, name: str = "source.png", size=(32, 24)) -> Path:
        path = self.root / name
        image = Image.new("RGBA", size, (20, 40, 60, 128))
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Comment", "remove me")
        image.save(path, pnginfo=metadata)
        return path

    def test_pixiv_defaults(self) -> None:
        self.assertEqual(pixiv_block_size(399, 200), 4)
        self.assertEqual(pixiv_block_size(1000, 400), 10)
        self.assertEqual(pixiv_brush_size(10), 45)
        self.assertEqual(scaled_mosaic_block_size(9, 2, 3), 6)
        self.assertEqual(scaled_mosaic_block_size(9, 1, 2), 4)
        self.assertEqual(scaled_mosaic_block_size(1, 1, 2), 1)
        self.assertEqual(scaled_mosaic_block_size(200, 1, 1), 50)

    def test_rejects_non_png_and_fake_png(self) -> None:
        processor = ImageProcessor()
        jpg = self.root / "image.jpg"
        Image.new("RGB", (4, 4)).save(jpg)
        with self.assertRaises(ImageLoadError):
            processor.load_png(jpg)
        fake = self.root / "fake.png"
        fake.write_text("not an image", encoding="utf-8")
        with self.assertRaises(ImageLoadError):
            processor.load_png(fake)

    def test_stroke_history_and_eraser(self) -> None:
        processor = ImageProcessor()
        processor.load_png(self.make_png())
        history = HistoryStack()
        processor.begin_stroke(ToolType.MOSAIC, 8)
        processor.add_stroke_point(10, 10)
        patch = processor.commit_stroke()
        self.assertIsNotNone(patch)
        history.push(patch)
        self.assertIsNotNone(processor.mosaic_mask.getbbox())
        self.assertTrue(history.undo(processor))
        self.assertIsNone(processor.mosaic_mask.getbbox())
        self.assertTrue(history.redo(processor))
        self.assertIsNotNone(processor.mosaic_mask.getbbox())

        processor.begin_stroke(ToolType.ERASER, 12)
        processor.add_stroke_point(10, 10)
        processor.commit_stroke()
        self.assertEqual(processor.mosaic_mask.getpixel((10, 10)), 0)

    def test_new_edit_discards_redo_and_clear_is_undoable(self) -> None:
        processor = ImageProcessor()
        processor.load_png(self.make_png())
        history = HistoryStack()
        processor.begin_stroke(ToolType.FILL, 6, (255, 255, 255))
        processor.add_stroke_point(6, 6)
        history.push(processor.commit_stroke())
        self.assertTrue(history.undo(processor))

        processor.begin_stroke(ToolType.MOSAIC, 6)
        processor.add_stroke_point(20, 12)
        history.push(processor.commit_stroke())
        self.assertFalse(history.can_redo)
        clear_patch = processor.clear_layers()
        self.assertIsNotNone(clear_patch)
        history.push(clear_patch)
        self.assertIsNone(processor.mosaic_mask.getbbox())
        self.assertTrue(history.undo(processor))
        self.assertIsNotNone(processor.mosaic_mask.getbbox())

    def test_mask_dilation_expands_by_neighboring_block(self) -> None:
        mask = Image.new("L", (20, 20), 0)
        mask.putpixel((10, 10), 255)
        expanded = ImageProcessor._dilate_mask_by_block(mask, 4)
        self.assertEqual(expanded.getpixel((10, 10)), 255)
        self.assertEqual(expanded.getpixel((6, 10)), 255)
        self.assertEqual(expanded.getpixel((18, 10)), 0)

    def test_mask_dilation_expands_by_selected_pixels(self) -> None:
        mask = Image.new("L", (20, 20), 0)
        mask.putpixel((10, 10), 255)
        expanded = ImageProcessor._dilate_mask_by_pixels(mask, 2)
        self.assertEqual(expanded.getpixel((8, 10)), 255)
        self.assertEqual(expanded.getpixel((7, 10)), 0)

    def test_fill_is_above_mosaic_and_alpha_is_preserved(self) -> None:
        processor = ImageProcessor()
        processor.load_png(self.make_png())
        processor.begin_stroke(ToolType.MOSAIC, 12)
        processor.add_stroke_point(12, 12)
        processor.commit_stroke()
        processor.begin_stroke(ToolType.FILL, 6, (255, 0, 0))
        processor.add_stroke_point(12, 12)
        processor.commit_stroke()
        result = processor.render_full(4, 1)
        self.assertEqual(result.getpixel((12, 12)), (255, 0, 0, 128))
        self.assertEqual(result.getchannel("A").tobytes(), processor.original_image.getchannel("A").tobytes())

    def test_saved_png_has_no_input_metadata(self) -> None:
        processor = ImageProcessor()
        source = self.make_png()
        processor.load_png(source)
        destination = self.root / "output.png"
        data = processor.encode_png(4, 1)
        processor.save_encoded_png(destination, data)
        with Image.open(destination) as saved:
            self.assertEqual(saved.format, "PNG")
            self.assertEqual(saved.mode, "RGBA")
            self.assertNotIn("Comment", saved.info)
            self.assertNotIn("exif", saved.info)
            self.assertNotIn("icc_profile", saved.info)

    def test_preview_caches_are_reused_and_invalidated(self) -> None:
        processor = ImageProcessor()
        processor.load_png(self.make_png(size=(2000, 1000)))
        self.assertEqual(processor.preview_original.size, (1600, 800))
        processor.render_preview(20, 1)
        first_layers = processor._preview_layers_cache
        first_pixelated = processor._preview_pixelated_cache[16]
        processor.render_preview(20, 1)
        self.assertIs(processor._preview_layers_cache, first_layers)
        self.assertIs(processor._preview_pixelated_cache[16], first_pixelated)

        processor.begin_stroke(ToolType.MOSAIC, 30)
        processor.add_stroke_point(100, 100)
        processor.commit_stroke()
        self.assertIsNone(processor._preview_layers_cache)

    def test_source_cannot_be_overwritten(self) -> None:
        processor = ImageProcessor()
        source = self.make_png()
        processor.load_png(source)
        with self.assertRaises(OriginalOverwriteError):
            processor.save_encoded_png(source, processor.encode_png(4, 1))

    def test_shape_mask_can_be_applied_and_undone(self) -> None:
        processor = ImageProcessor()
        processor.load_png(self.make_png())
        mask = Image.new("L", processor.size, 0)
        mask.putpixel((5, 5), 255)
        patch = processor.apply_mosaic_mask(mask)
        self.assertIsNotNone(patch)
        self.assertEqual(processor.mosaic_mask.getpixel((5, 5)), 255)
        patch.apply(processor, after=False)
        self.assertEqual(processor.mosaic_mask.getpixel((5, 5)), 0)


if __name__ == "__main__":
    unittest.main()
