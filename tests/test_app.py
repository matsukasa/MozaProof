from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF, QMimeData, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QSpinBox

from src.app import MainWindow
from src.settings import pixiv_block_size, pixiv_brush_size
from src.tools import ToolType


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.source = Path(self.temp_dir.name) / "sample.png"
        Image.new("RGBA", (800, 400), (30, 60, 90, 180)).save(self.source)
        self.window = MainWindow()
        self.window.auto_detect_checkbox.setChecked(False)

    def tearDown(self) -> None:
        self.window._dirty = False
        self.window.close()
        self.window.deleteLater()
        self.temp_dir.cleanup()

    def test_load_sets_pixiv_defaults_and_canvas(self) -> None:
        self.window.load_image(str(self.source))
        block = pixiv_block_size(800, 400)
        self.assertEqual(self.window.brush.mosaic_block_size, block)
        self.assertEqual(self.window.brush.size, pixiv_brush_size(block))
        self.assertFalse(self.window.brush.dilate_mosaic_mask)
        self.assertFalse(self.window.dilate_checkbox.isChecked())
        self.assertEqual(self.window.brush.tool, ToolType.MOSAIC)
        self.assertTrue(self.window.canvas.has_image())
        self.assertTrue(self.window.save_action.isEnabled())
        self.assertEqual(self.window.image_dimensions_label.text(), "800 × 400 px")
        self.assertEqual(self.window.image_long_edge_label.text(), "800 px")
        self.assertEqual(self.window.image_alpha_label.text(), "透過あり")
        self.assertEqual(self.window.image_recommended_mosaic_label.text(), "8 px")
        self.assertNotEqual(self.window.image_file_size_label.text(), "—")

    def test_stroke_undo_and_redo_through_window(self) -> None:
        self.window.load_image(str(self.source))
        self.window._begin_stroke(100, 100)
        self.window._continue_stroke(160, 100)
        self.window._end_stroke()
        self.assertTrue(self.window.history.can_undo)
        self.assertIsNotNone(self.window.processor.mosaic_mask.getbbox())
        self.window.undo()
        self.assertIsNone(self.window.processor.mosaic_mask.getbbox())
        self.window.redo()
        self.assertIsNotNone(self.window.processor.mosaic_mask.getbbox())

    def test_numeric_inputs_update_slider_and_brush_settings(self) -> None:
        self.assertIsInstance(self.window.brush_value, QSpinBox)
        self.assertIsInstance(self.window.block_value, QSpinBox)
        self.window.brush_value.setValue(137)
        self.window.block_value.setValue(17)
        self.assertEqual(self.window.brush_slider.value(), 137)
        self.assertEqual(self.window.block_slider.value(), 17)
        self.assertEqual(self.window.brush.size, 137)
        self.assertEqual(self.window.brush.mosaic_block_size, 17)

    def test_png_drop_loads_from_window_and_canvas_accepts_drops(self) -> None:
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(self.source))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.window.dropEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(self.window.processor.source_path, self.source.resolve())
        self.assertTrue(self.window.canvas.acceptDrops())
        self.assertTrue(self.window.canvas.viewport().acceptDrops())

    def test_auto_detection_result_is_undoable(self) -> None:
        from PIL import ImageDraw
        from src.auto_censor import AutoCensorResult

        self.window.load_image(str(self.source))
        mask = Image.new("L", (800, 400), 0)
        ImageDraw.Draw(mask).ellipse((100, 100, 180, 170), fill=255)
        self.window._auto_generation = 7
        self.window._auto_detection_finished(7, AutoCensorResult(mask, 1, 1))
        self.assertIsNotNone(self.window.processor.mosaic_mask.getbbox())
        self.assertTrue(self.window.history.can_undo)
        self.window.undo()
        self.assertIsNone(self.window.processor.mosaic_mask.getbbox())


if __name__ == "__main__":
    unittest.main()
