import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from src.image_processor import ImageProcessor


class PngSettingsTests(unittest.TestCase):
    def test_preserve_or_strip_metadata_at_each_compression_level(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            image = Image.new("RGBA", (160, 80), (15, 40, 90, 70))
            info = PngImagePlugin.PngInfo()
            info.add_text("prompt", "original", zip=True)
            info.add_itxt("workflow", "日本語", lang="ja", zip=True)
            info.add(b"gAMA", (45455).to_bytes(4, "big"))
            info.add(b"vpAg", b"unknown private chunk", after_idat=True)
            exif = Image.Exif()
            exif[315] = "artist"
            image.save(path, pnginfo=info, exif=exif, icc_profile=b"profile", dpi=(96, 96))
            processor = ImageProcessor()
            processor.load_png(path)
            self.assertEqual(processor.unsupported_png_chunks, ("vpAg",))
            sizes = {}
            for level in (0, 1, 9):
                for keep in (False, True):
                    with self.subTest(level=level, keep=keep):
                        png = processor.encode_png(4, compress_level=level, keep_metadata=keep)
                        sizes[level, keep] = len(png)
                        with Image.open(io.BytesIO(png)) as output:
                            self.assertEqual(output.tobytes(), image.tobytes())
                            self.assertEqual(output.text, {"prompt": "original", "workflow": "日本語"} if keep else {})
                            self.assertEqual(output.getexif().get(315), "artist" if keep else None)
                            self.assertEqual(output.info.get("icc_profile"), b"profile" if keep else None)
                            self.assertEqual(output.info.get("gamma"), 0.45455 if keep else None)
                            self.assertEqual("dpi" in output.info, keep)
                            if keep:
                                self.assertEqual(output.text["workflow"].lang, "ja")
                        self.assertNotIn(b"vpAg", png)
            self.assertGreater(sizes[0, False], sizes[9, False])
            clean = Path(directory) / "clean.png"
            image.save(clean)
            processor.load_png(clean)
            self.assertEqual(processor.unsupported_png_chunks, ())
            with Image.open(io.BytesIO(processor.encode_png(4, keep_metadata=True))) as output:
                self.assertEqual(output.info, {})
