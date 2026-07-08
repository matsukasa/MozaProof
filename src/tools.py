from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .settings import DEFAULT_FILL_COLOR


class ToolType(Enum):
    MOSAIC = auto()
    FILL = auto()
    ERASER = auto()
    PAN = auto()


@dataclass(slots=True)
class BrushSettings:
    tool: ToolType = ToolType.MOSAIC
    size: int = 30
    mosaic_block_size: int = 4
    fill_color: tuple[int, int, int] = DEFAULT_FILL_COLOR
    dilate_mosaic_mask: bool = False
