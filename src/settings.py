from __future__ import annotations

APP_NAME = "MozaProof"
PREVIEW_MAX_EDGE = 1600
PNG_SIZE_WARNING_BYTES = 32 * 1024 * 1024
MIN_MOSAIC_BLOCK_SIZE = 4
MAX_MOSAIC_BLOCK_SIZE = 50
DEFAULT_BRUSH_SIZE = 45
MAX_BRUSH_SIZE = 100
DEFAULT_MASK_DILATION_PX = 1
MAX_MASK_DILATION_PX = 10
DEFAULT_FILL_COLOR = (0, 0, 0)
MOSAIC_SCALE_OPTIONS = (
    ("標準", 1, 1),
    ("2/3", 2, 3),
    ("1/2", 1, 2),
)


def pixiv_block_size(width: int, height: int) -> int:
    """Return the requested pixiv-oriented block size for an image."""
    return max(MIN_MOSAIC_BLOCK_SIZE, round(max(width, height) / 100))


def pixiv_brush_size(block_size: int) -> int:
    del block_size
    return DEFAULT_BRUSH_SIZE


def scaled_mosaic_block_size(block_size: int, numerator: int, denominator: int) -> int:
    """Return a rounded block size after applying a user-selected scale."""
    return min(MAX_MOSAIC_BLOCK_SIZE, max(1, round(block_size * numerator / denominator)))
