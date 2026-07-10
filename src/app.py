from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import QSignalBlocker, QThread, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QCloseEvent, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .auto_censor import AutoCensorResult, AutoCensorWorker
from .history import HistoryStack
from .image_processor import ImageLoadError, ImageProcessor, OriginalOverwriteError
from .settings import (
    APP_NAME,
    DEFAULT_BRUSH_SIZE,
    DEFAULT_MASK_DILATION_PX,
    MAX_BRUSH_SIZE,
    MAX_MASK_DILATION_PX,
    MAX_MOSAIC_BLOCK_SIZE,
    MOSAIC_SCALE_OPTIONS,
    PNG_SIZE_WARNING_BYTES,
    pixiv_block_size,
    pixiv_brush_size,
    scaled_mosaic_block_size,
)
from .tools import BrushSettings, ToolType


def resource_path(relative_path: str) -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / relative_path
    return Path(__file__).resolve().parents[1] / relative_path


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        icon_path = resource_path("assets/app_icon_girl.ico")
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.processor = ImageProcessor()
        self.history = HistoryStack()
        self.brush = BrushSettings()
        self._dirty = False
        self._comparing = False
        self._fill_color = QColor(0, 0, 0)
        self._auto_generation = 0
        self._auto_thread: QThread | None = None
        self._auto_worker: AutoCensorWorker | None = None
        self._pending_auto_request: tuple[int, str] | None = None
        self._base_mosaic_block_size = 4

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(50)
        self._preview_timer.timeout.connect(self._refresh_preview)

        self._build_ui()
        self._connect_canvas()
        self._update_enabled_state()

    def _build_ui(self) -> None:
        toolbar = QToolBar("メイン", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.open_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "画像を開く",
            self,
        )
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_image_dialog)
        toolbar.addAction(self.open_action)

        self.save_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "名前を付けて保存",
            self,
        )
        self.save_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_action.triggered.connect(self.save_as)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()

        self.undo_action = QAction("元に戻す", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        toolbar.addAction(self.undo_action)

        self.redo_action = QAction("やり直す", self)
        self.redo_action.setShortcuts(
            [QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")]
        )
        self.redo_action.triggered.connect(self.redo)
        toolbar.addAction(self.redo_action)

        self.clear_action = QAction("全加工をクリア", self)
        self.clear_action.triggered.connect(self.clear_layers)
        toolbar.addAction(self.clear_action)

        self.canvas = ImageCanvas(self)
        self.side_panel = self._create_side_panel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.side_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([940, 260])
        self.setCentralWidget(splitter)

        self.zoom_label = QLabel("ズーム: --")
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.statusBar().showMessage("PNG画像を開くか、キャンバスへドロップしてください。")

    def _create_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info_group = QGroupBox("画像情報")
        info_layout = QFormLayout(info_group)
        self.image_dimensions_label = QLabel("—")
        self.image_long_edge_label = QLabel("—")
        self.image_file_size_label = QLabel("—")
        self.image_alpha_label = QLabel("—")
        self.image_recommended_mosaic_label = QLabel("—")
        info_layout.addRow("画像サイズ", self.image_dimensions_label)
        info_layout.addRow("長辺", self.image_long_edge_label)
        info_layout.addRow("元PNG容量", self.image_file_size_label)
        info_layout.addRow("透明度", self.image_alpha_label)
        info_layout.addRow("推奨モザイク", self.image_recommended_mosaic_label)
        layout.addWidget(info_group)

        auto_group = QGroupBox("自動モザイク")
        auto_layout = QVBoxLayout(auto_group)
        self.auto_detect_checkbox = QCheckBox("読み込み時に性器を自動検出")
        self.auto_detect_checkbox.setChecked(True)
        self.auto_detect_checkbox.toggled.connect(self._auto_detection_toggled)
        self.auto_detect_button = QPushButton("自動検出を再実行")
        self.auto_detect_button.clicked.connect(self._rerun_auto_detection)
        self.auto_detect_status = QLabel("画像読み込み後に実行します。")
        self.auto_detect_status.setWordWrap(True)
        auto_layout.addWidget(self.auto_detect_checkbox)
        auto_layout.addWidget(self.auto_detect_button)
        auto_layout.addWidget(self.auto_detect_status)
        layout.addWidget(auto_group)

        tools_group = QGroupBox("ツール")
        tools_layout = QVBoxLayout(tools_group)
        self.tool_buttons: dict[ToolType, QRadioButton] = {}
        for tool, label in (
            (ToolType.MOSAIC, "モザイクブラシ"),
            (ToolType.FILL, "塗りつぶしブラシ"),
            (ToolType.ERASER, "モザイクを解除"),
        ):
            button = QRadioButton(label)
            button.toggled.connect(
                lambda checked, selected=tool: checked and self._select_tool(selected)
            )
            tools_layout.addWidget(button)
            self.tool_buttons[tool] = button
        self.tool_buttons[ToolType.MOSAIC].setChecked(True)
        layout.addWidget(tools_group)

        settings_group = QGroupBox("ブラシ設定")
        settings_layout = QFormLayout(settings_group)
        self.brush_slider, self.brush_value = self._make_slider(
            1, MAX_BRUSH_SIZE, DEFAULT_BRUSH_SIZE
        )
        self.brush_slider.valueChanged.connect(self._brush_size_changed)
        settings_layout.addRow("ブラシサイズ", self._slider_row(self.brush_slider, self.brush_value))

        self.block_slider, self.block_value = self._make_slider(
            1, MAX_MOSAIC_BLOCK_SIZE, 4
        )
        self.block_slider.valueChanged.connect(self._block_size_changed)
        self.block_scale_combo = QComboBox()
        for label, numerator, denominator in MOSAIC_SCALE_OPTIONS:
            self.block_scale_combo.addItem(label, (numerator, denominator))
        self.block_scale_combo.currentIndexChanged.connect(self._block_scale_changed)
        settings_layout.addRow("モザイク倍率", self.block_scale_combo)
        settings_layout.addRow("ブロックサイズ", self._slider_row(self.block_slider, self.block_value))

        self.dilate_checkbox = QCheckBox("マスクをピクセル単位で膨張")
        self.dilate_checkbox.setChecked(False)
        self.dilate_checkbox.toggled.connect(self._processing_setting_changed)
        settings_layout.addRow(self.dilate_checkbox)
        self.dilation_value = QSpinBox()
        self.dilation_value.setRange(1, MAX_MASK_DILATION_PX)
        self.dilation_value.setValue(DEFAULT_MASK_DILATION_PX)
        self.dilation_value.setSuffix(" px")
        self.dilation_value.setEnabled(False)
        self.dilation_value.valueChanged.connect(self._processing_setting_changed)
        settings_layout.addRow("膨張量", self.dilation_value)

        self.reset_button = QPushButton("pixiv推奨値に戻す")
        self.reset_button.clicked.connect(self.reset_pixiv_values)
        settings_layout.addRow(self.reset_button)
        layout.addWidget(settings_group)

        colors_group = QGroupBox("塗りつぶし色")
        colors_layout = QHBoxLayout(colors_group)
        black = QPushButton("黒")
        black.clicked.connect(lambda: self._set_fill_color(QColor(0, 0, 0)))
        white = QPushButton("白")
        white.clicked.connect(lambda: self._set_fill_color(QColor(255, 255, 255)))
        self.custom_color_button = QPushButton("任意色…")
        self.custom_color_button.clicked.connect(self.choose_color)
        colors_layout.addWidget(black)
        colors_layout.addWidget(white)
        colors_layout.addWidget(self.custom_color_button)
        layout.addWidget(colors_group)

        self.compare_button = QPushButton("押している間、加工前を表示")
        self.compare_button.pressed.connect(self._show_original)
        self.compare_button.released.connect(self._show_processed)
        layout.addWidget(self.compare_button)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _make_slider(minimum: int, maximum: int, value: int) -> tuple[QSlider, QSpinBox]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        number_input = QSpinBox()
        number_input.setRange(minimum, maximum)
        number_input.setValue(value)
        number_input.setSuffix(" px")
        number_input.setKeyboardTracking(False)
        number_input.setMinimumWidth(82)
        slider.valueChanged.connect(number_input.setValue)
        number_input.valueChanged.connect(slider.setValue)
        return slider, number_input

    @staticmethod
    def _slider_row(slider: QSlider, number_input: QSpinBox) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(number_input)
        return widget

    def _connect_canvas(self) -> None:
        self.canvas.stroke_began.connect(self._begin_stroke)
        self.canvas.stroke_moved.connect(self._continue_stroke)
        self.canvas.stroke_ended.connect(self._end_stroke)
        self.canvas.files_dropped.connect(self._handle_drop)
        self.canvas.zoom_changed.connect(
            lambda value: self.zoom_label.setText(f"ズーム: {value}%")
        )

    def _select_tool(self, tool: ToolType) -> None:
        self.brush.tool = tool
        self.canvas.set_tool(tool)

    def _brush_size_changed(self, value: int) -> None:
        self.brush.size = value
        self.canvas.set_brush_size(value)

    def _block_size_changed(self, value: int) -> None:
        self.brush.mosaic_block_size = value
        if self.processor.is_loaded:
            self._dirty = True
            self._schedule_preview()

    def _block_scale_changed(self) -> None:
        block = self._scaled_recommended_block_size()
        self._set_input_maximum(
            self.block_slider, self.block_value, MAX_MOSAIC_BLOCK_SIZE
        )
        self.block_slider.setValue(block)

    def _processing_setting_changed(self, *_args) -> None:
        self.brush.dilate_mosaic_mask = self.dilate_checkbox.isChecked()
        self.brush.mosaic_mask_dilation_px = self.dilation_value.value()
        self.dilation_value.setEnabled(self.brush.dilate_mosaic_mask)
        if self.processor.is_loaded:
            self._dirty = True
            self._schedule_preview()

    def reset_pixiv_values(self) -> None:
        if not self.processor.is_loaded:
            return
        width, height = self.processor.size
        self._base_mosaic_block_size = pixiv_block_size(width, height)
        self.block_scale_combo.setCurrentIndex(0)
        block = self._scaled_recommended_block_size()
        brush = pixiv_brush_size(block)
        self._set_input_maximum(
            self.block_slider, self.block_value, MAX_MOSAIC_BLOCK_SIZE
        )
        self._set_input_maximum(self.brush_slider, self.brush_value, MAX_BRUSH_SIZE)
        self.block_slider.setValue(block)
        self.brush_slider.setValue(brush)
        self.dilate_checkbox.setChecked(False)
        self.dilation_value.setValue(DEFAULT_MASK_DILATION_PX)

    def _scaled_recommended_block_size(self) -> int:
        numerator, denominator = self.block_scale_combo.currentData() or (1, 1)
        return scaled_mosaic_block_size(
            self._base_mosaic_block_size, numerator, denominator
        )

    def _mask_dilation_px(self) -> int:
        if not self.brush.dilate_mosaic_mask:
            return 0
        return min(MAX_MASK_DILATION_PX, max(1, self.brush.mosaic_mask_dilation_px))

    def choose_color(self) -> None:
        color = QColorDialog.getColor(self._fill_color, self, "塗りつぶし色を選択")
        if color.isValid():
            self._set_fill_color(color)

    def _set_fill_color(self, color: QColor) -> None:
        self._fill_color = color
        self.brush.fill_color = (color.red(), color.green(), color.blue())
        self.custom_color_button.setStyleSheet(
            f"background-color: {color.name()}; color: "
            f"{'black' if color.lightness() > 128 else 'white'};"
        )

    def open_image_dialog(self) -> None:
        if not self._confirm_discard_for_new_image():
            return
        path, _ = QFileDialog.getOpenFileName(self, "PNG画像を開く", "", "PNG画像 (*.png)")
        if path:
            self.load_image(path)

    def _handle_drop(self, paths: list[str]) -> None:
        if len(paths) != 1:
            QMessageBox.warning(self, APP_NAME, "PNG画像を1ファイルだけドロップしてください。")
            return
        if Path(paths[0]).suffix.lower() != ".png":
            QMessageBox.warning(self, APP_NAME, "PNGのみ対応しています。")
            return
        if self._confirm_discard_for_new_image():
            self.load_image(paths[0])

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        self._handle_drop(paths)
        event.acceptProposedAction()

    def load_image(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.processor.load_png(path)
        except ImageLoadError as exc:
            QMessageBox.critical(self, "読み込みエラー", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.history.clear()
        self._dirty = False
        with (
            QSignalBlocker(self.block_slider),
            QSignalBlocker(self.brush_slider),
            QSignalBlocker(self.block_value),
            QSignalBlocker(self.brush_value),
            QSignalBlocker(self.block_scale_combo),
            QSignalBlocker(self.dilate_checkbox),
            QSignalBlocker(self.dilation_value),
        ):
            width, height = self.processor.size
            self._base_mosaic_block_size = pixiv_block_size(width, height)
            self.block_scale_combo.setCurrentIndex(0)
            block = self._scaled_recommended_block_size()
            brush = pixiv_brush_size(block)
            self._set_input_maximum(
                self.block_slider, self.block_value, MAX_MOSAIC_BLOCK_SIZE
            )
            self._set_input_maximum(
                self.brush_slider, self.brush_value, MAX_BRUSH_SIZE
            )
            self.block_slider.setValue(block)
            self.brush_slider.setValue(brush)
            self.block_value.setValue(block)
            self.brush_value.setValue(brush)
            self.brush.mosaic_block_size = block
            self.brush.size = brush
            self.dilate_checkbox.setChecked(False)
            self.dilation_value.setValue(DEFAULT_MASK_DILATION_PX)
            self.dilation_value.setEnabled(False)
            self.brush.dilate_mosaic_mask = False
            self.brush.mosaic_mask_dilation_px = DEFAULT_MASK_DILATION_PX
        self.canvas.set_brush_size(self.brush.size)
        self.tool_buttons[ToolType.MOSAIC].setChecked(True)
        self._refresh_preview(fit=True)
        self._update_enabled_state()
        self.statusBar().showMessage(
            f"{Path(path).name} — {self.processor.size[0]} × {self.processor.size[1]} px"
        )
        self._update_image_info()
        self._auto_generation += 1
        self._pending_auto_request = None
        if self.auto_detect_checkbox.isChecked():
            QTimer.singleShot(0, self._queue_auto_detection)
        else:
            self.auto_detect_status.setText("自動検出はOFFです。")

    def _update_image_info(self) -> None:
        width, height = self.processor.size
        recommended = pixiv_block_size(width, height)
        self.image_dimensions_label.setText(f"{width} × {height} px")
        self.image_long_edge_label.setText(f"{max(width, height)} px")
        self.image_file_size_label.setText(
            self._format_file_size(self.processor.source_file_size)
        )
        self.image_alpha_label.setText(
            "透過あり" if self.processor.has_transparency else "透過なし"
        )
        self.image_recommended_mosaic_label.setText(f"{recommended} px")

    @staticmethod
    def _format_file_size(size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    @staticmethod
    def _set_input_maximum(
        slider: QSlider, number_input: QSpinBox, maximum: int
    ) -> None:
        slider.setMaximum(maximum)
        number_input.setMaximum(maximum)

    def _auto_detection_toggled(self, checked: bool) -> None:
        if not checked:
            self._auto_generation += 1
            self._pending_auto_request = None
            if self._auto_thread and self._auto_thread.isRunning():
                self._auto_thread.requestInterruption()
            self.auto_detect_status.setText("自動検出はOFFです。")
        elif self.processor.is_loaded:
            self._auto_generation += 1
            self._queue_auto_detection()
        self._update_enabled_state()

    def _rerun_auto_detection(self) -> None:
        if not self.processor.is_loaded:
            return
        self._auto_generation += 1
        self._queue_auto_detection()

    def _queue_auto_detection(self) -> None:
        if not self.processor.is_loaded or self.processor.source_path is None:
            return
        request = (self._auto_generation, str(self.processor.source_path))
        if self._auto_thread and self._auto_thread.isRunning():
            self._pending_auto_request = request
            self._auto_thread.requestInterruption()
            self.auto_detect_status.setText("前の検出終了後に実行します…")
            return
        self._launch_auto_detection(*request)

    def _launch_auto_detection(self, generation: int, image_path: str) -> None:
        thread = QThread(self)
        worker = AutoCensorWorker(generation, image_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._auto_detection_finished)
        worker.failed.connect(self._auto_detection_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda selected=thread: self._auto_thread_finished(selected)
        )
        self._auto_thread = thread
        self._auto_worker = worker
        self.auto_detect_button.setEnabled(False)
        self.auto_detect_status.setText("性器を自動検出中…")
        thread.start()

    def _auto_detection_finished(
        self, generation: int, result: AutoCensorResult
    ) -> None:
        if generation != self._auto_generation or not self.processor.is_loaded:
            return
        patch = self.processor.apply_mosaic_mask(result.mask)
        if patch is not None:
            self.history.push(patch)
            self._dirty = True
            self._refresh_preview()
            self._update_enabled_state()
        if result.detection_count:
            self.auto_detect_status.setText(
                f"{result.segmented_count}箇所を自動処理しました。必ず確認してください。"
            )
        else:
            self.auto_detect_status.setText(
                "候補は見つかりませんでした。見逃しがないか確認してください。"
            )

    def _auto_detection_failed(self, generation: int, message: str) -> None:
        if generation == self._auto_generation:
            self.auto_detect_status.setText(f"自動検出に失敗しました: {message}")

    def _auto_thread_finished(self, thread: QThread) -> None:
        if self._auto_thread is thread:
            self._auto_thread = None
            self._auto_worker = None
            self._update_enabled_state()
        pending = self._pending_auto_request
        self._pending_auto_request = None
        if pending and pending[0] == self._auto_generation:
            self._launch_auto_detection(*pending)

    def _begin_stroke(self, x: float, y: float) -> None:
        self.processor.begin_stroke(
            self.brush.tool, self.brush.size, self.brush.fill_color
        )
        self.processor.add_stroke_point(x, y)
        self._schedule_preview()

    def _continue_stroke(self, x: float, y: float) -> None:
        self.processor.add_stroke_point(x, y)
        self._schedule_preview()

    def _end_stroke(self) -> None:
        patch = self.processor.commit_stroke()
        if patch is not None:
            self.history.push(patch)
            self._dirty = True
        self._refresh_preview()
        self._update_enabled_state()

    def _schedule_preview(self) -> None:
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _pil_pixmap(self, image) -> QPixmap:
        return QPixmap.fromImage(ImageQt(image).copy())

    def _refresh_preview(self, fit: bool = False) -> None:
        if not self.processor.is_loaded or self._comparing:
            return
        image = self.processor.render_preview(
            self.brush.mosaic_block_size, self._mask_dilation_px()
        )
        self.canvas.set_preview(
            self._pil_pixmap(image), self.processor.size, fit=fit
        )

    def _show_original(self) -> None:
        if not self.processor.is_loaded or self.processor.preview_original is None:
            return
        self._comparing = True
        self.canvas.set_preview(
            self._pil_pixmap(self.processor.preview_original), self.processor.size
        )

    def _show_processed(self) -> None:
        if not self.processor.is_loaded:
            return
        self._comparing = False
        self._refresh_preview()

    def undo(self) -> None:
        if self.history.undo(self.processor):
            self._dirty = True
            self._refresh_preview()
            self._update_enabled_state()

    def redo(self) -> None:
        if self.history.redo(self.processor):
            self._dirty = True
            self._refresh_preview()
            self._update_enabled_state()

    def clear_layers(self) -> None:
        patch = self.processor.clear_layers()
        if patch is not None:
            self.history.push(patch)
            self._dirty = True
            self._refresh_preview()
            self._update_enabled_state()

    def save_as(self) -> None:
        if not self.processor.is_loaded or self.processor.source_path is None:
            return
        default_name = f"{self.processor.source_path.stem}_pixiv_safe.png"
        initial = str(self.processor.source_path.with_name(default_name))
        while True:
            path, _ = QFileDialog.getSaveFileName(
                self, "名前を付けて保存", initial, "PNG画像 (*.png)"
            )
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            if self._same_as_source(path):
                QMessageBox.warning(
                    self,
                    "元画像は上書きできません",
                    "元画像を保護するため、別のファイル名を指定してください。",
                )
                initial = path
                continue
            break

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data = self.processor.encode_png(
                self.brush.mosaic_block_size, self._mask_dilation_px()
            )
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.critical(self, "保存エラー", f"PNGの生成に失敗しました。\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if len(data) > PNG_SIZE_WARNING_BYTES:
            size_mb = len(data) / (1024 * 1024)
            answer = QMessageBox.warning(
                self,
                "ファイルサイズの警告",
                f"保存後のPNGは約 {size_mb:.1f} MB です。\n32 MBを超えますが保存しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            self.processor.save_encoded_png(path, data)
        except (OSError, ValueError, OriginalOverwriteError) as exc:
            QMessageBox.critical(self, "保存エラー", f"保存できませんでした。\n{exc}")
            return
        self._dirty = False
        QMessageBox.information(self, "保存完了", f"PNGを保存しました。\n{path}")

    def _same_as_source(self, path: str) -> bool:
        if self.processor.source_path is None:
            return False
        return os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            os.path.abspath(self.processor.source_path)
        )

    def _update_enabled_state(self) -> None:
        loaded = self.processor.is_loaded
        detecting = bool(self._auto_thread and self._auto_thread.isRunning())
        self.save_action.setEnabled(loaded)
        self.undo_action.setEnabled(loaded and self.history.can_undo)
        self.redo_action.setEnabled(loaded and self.history.can_redo)
        self.clear_action.setEnabled(loaded)
        self.side_panel.setEnabled(loaded)
        self.auto_detect_button.setEnabled(
            loaded and self.auto_detect_checkbox.isChecked() and not detecting
        )

    def _confirm_discard_for_new_image(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "別の画像を開きますか？",
            "現在の画像に未保存の加工があります。\n"
            "保存せずにこの加工を破棄して、別の画像を開きますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_discard_for_close(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "アプリを終了しますか？",
            "未保存の加工があります。\n"
            "保存せずにこの加工を破棄して、アプリを終了しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_discard_for_close():
            if self._auto_thread and self._auto_thread.isRunning():
                self._auto_thread.requestInterruption()
                self._auto_thread.quit()
                if not self._auto_thread.wait(10000):
                    self.statusBar().showMessage("自動検出の終了を待っています…")
                    event.ignore()
                    return
            event.accept()
        else:
            event.ignore()


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()
