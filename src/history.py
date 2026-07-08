from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .image_processor import ImageProcessor


@dataclass
class LayerPatch:
    bbox: tuple[int, int, int, int]
    before_mosaic: Image.Image
    after_mosaic: Image.Image
    before_fill: Image.Image
    after_fill: Image.Image

    def apply(self, processor: "ImageProcessor", *, after: bool) -> None:
        mosaic = self.after_mosaic if after else self.before_mosaic
        fill = self.after_fill if after else self.before_fill
        processor.mosaic_mask.paste(mosaic, self.bbox[:2])
        processor.fill_layer.paste(fill, self.bbox[:2])
        processor.invalidate_preview_layers()


class HistoryStack:
    """Small command stack that stores only changed image rectangles."""

    def __init__(self) -> None:
        self._commands: list[LayerPatch] = []
        self._index = 0

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._commands)

    @property
    def index(self) -> int:
        return self._index

    def clear(self) -> None:
        self._commands.clear()
        self._index = 0

    def push(self, command: LayerPatch) -> None:
        del self._commands[self._index :]
        self._commands.append(command)
        self._index += 1

    def undo(self, processor: "ImageProcessor") -> bool:
        if not self.can_undo:
            return False
        self._index -= 1
        self._commands[self._index].apply(processor, after=False)
        return True

    def redo(self, processor: "ImageProcessor") -> bool:
        if not self.can_redo:
            return False
        self._commands[self._index].apply(processor, after=True)
        self._index += 1
        return True
