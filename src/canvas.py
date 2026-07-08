from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .tools import ToolType


class ImageCanvas(QGraphicsView):
    stroke_began = Signal(float, float)
    stroke_moved = Signal(float, float)
    stroke_ended = Signal()
    files_dropped = Signal(object)
    zoom_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor(45, 45, 48)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._cursor = QGraphicsEllipseItem()
        self._cursor.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
        self._cursor.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._cursor.setZValue(10)
        self._cursor.hide()
        self._scene.addItem(self._cursor)

        self._image_size = (0, 0)
        self._tool = ToolType.MOSAIC
        self._brush_size = 30
        self._painting = False
        self._panning = False
        self._space_down = False
        self._pan_start = QPoint()

    def has_image(self) -> bool:
        return self._image_size != (0, 0)

    def set_tool(self, tool: ToolType) -> None:
        self._tool = tool
        if tool is ToolType.PAN:
            self._cursor.hide()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, size)
        self._update_cursor(self.mapFromGlobal(self.cursor().pos()))

    def set_preview(
        self, pixmap: QPixmap, image_size: tuple[int, int], *, fit: bool = False
    ) -> None:
        self._image_size = image_size
        self._pixmap_item.setPixmap(pixmap)
        if pixmap.width() > 0:
            self._pixmap_item.setScale(image_size[0] / pixmap.width())
        self._scene.setSceneRect(QRectF(0, 0, image_size[0], image_size[1]))
        if fit:
            self.fit_image()

    def fit_image(self) -> None:
        if not self.has_image():
            return
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_zoom()

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(round(self.transform().m11() * 100))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image():
            return
        current = self.transform().m11()
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_scale = current * factor
        if 0.01 <= new_scale <= 64:
            self.scale(factor, factor)
            self._emit_zoom()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pan_requested = (
            event.button() == Qt.MouseButton.MiddleButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_down)
            or (event.button() == Qt.MouseButton.LeftButton and self._tool is ToolType.PAN)
        )
        if pan_requested:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.has_image()
            and self._tool is not ToolType.PAN
        ):
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self._painting = True
                self.stroke_began.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        self._update_cursor(event.position().toPoint())
        if self._painting:
            point = self.mapToScene(event.position().toPoint())
            rect = self.sceneRect()
            self.stroke_moved.emit(
                min(max(point.x(), rect.left()), rect.right() - 1),
                min(max(point.y(), rect.top()), rect.bottom() - 1),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        if self._painting and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self.stroke_ended.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _update_cursor(self, viewport_point: QPoint) -> None:
        if not self.has_image() or self._tool is ToolType.PAN or self._panning:
            self._cursor.hide()
            return
        point = self.mapToScene(viewport_point)
        if not self.sceneRect().contains(point):
            self._cursor.hide()
            return
        radius = self._brush_size / 2
        self._cursor.setRect(
            point.x() - radius,
            point.y() - radius,
            self._brush_size,
            self._brush_size,
        )
        self._cursor.show()

    def leaveEvent(self, event) -> None:
        self._cursor.hide()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            event.accept()
            return
        super().keyReleaseEvent(event)

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
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.files_dropped.emit(paths)
        event.acceptProposedAction()
