from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .settings import DEFAULT_BRUSH_SIZE, DEFAULT_FILL_COLOR, DEFAULT_MASK_DILATION_PX


class ToolType(Enum):
    MOSAIC = auto()
    FILL = auto()
    ERASER = auto()
    PAN = auto()


@dataclass(slots=True)
class BrushSettings:
    tool: ToolType = ToolType.MOSAIC
    size: int = DEFAULT_BRUSH_SIZE
    mosaic_block_size: int = 4
    fill_color: tuple[int, int, int] = DEFAULT_FILL_COLOR
    dilate_mosaic_mask: bool = False
    mosaic_mask_dilation_px: int = DEFAULT_MASK_DILATION_PX
