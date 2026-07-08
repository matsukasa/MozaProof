from __future__ import annotations

import io
import math
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .history import LayerPatch
from .settings import PREVIEW_MAX_EDGE
from .tools import ToolType


class ImageLoadError(ValueError):
    pass


class OriginalOverwriteError(ValueError):
    pass


class ImageProcessor:
    def __init__(self) -> None:
        self.source_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.mosaic_mask: Image.Image | None = None
        self.fill_layer: Image.Image | None = None
        self.preview_original: Image.Image | None = None
        self.preview_scale = 1.0
        self.source_file_size = 0
        self.has_transparency = False
        self._preview_layers_cache: tuple[Image.Image, Image.Image] | None = None
        self._preview_pixelated_cache: dict[int, Image.Image] = {}

        self._stroke_mask: Image.Image | None = None
        self._stroke_tool: ToolType | None = None
        self._stroke_color = (0, 0, 0)
        self._stroke_size = 1
        self._stroke_bbox: tuple[int, int, int, int] | None = None
        self._last_point: tuple[float, float] | None = None

    @property
    def is_loaded(self) -> bool:
        return self.original_image is not None

    @property
    def size(self) -> tuple[int, int]:
        if self.original_image is None:
            return (0, 0)
        return self.original_image.size

    def load_png(self, path: str | os.PathLike[str]) -> None:
        source = Path(path)
        if source.suffix.lower() != ".png":
            raise ImageLoadError("PNGのみ対応しています。")
        try:
            with Image.open(source) as opened:
                if opened.format != "PNG":
                    raise ImageLoadError("PNGのみ対応しています。")
                opened.load()
                original = opened.convert("RGBA")
        except ImageLoadError:
            raise
        except (OSError, ValueError) as exc:
            raise ImageLoadError(f"PNG画像を読み込めませんでした: {exc}") from exc

        self.source_path = source.resolve()
        self.original_image = original
        try:
            self.source_file_size = source.stat().st_size
        except OSError:
            self.source_file_size = 0
        self.has_transparency = original.getchannel("A").getextrema()[0] < 255
        self.mosaic_mask = Image.new("L", original.size, 0)
        self.fill_layer = Image.new("RGBA", original.size, (0, 0, 0, 0))
        self._make_preview_cache()
        self.cancel_stroke()

    def _make_preview_cache(self) -> None:
        assert self.original_image is not None
        width, height = self.original_image.size
        longest = max(width, height)
        self.preview_scale = min(1.0, PREVIEW_MAX_EDGE / longest)
        preview_size = (
            max(1, round(width * self.preview_scale)),
            max(1, round(height * self.preview_scale)),
        )
        self.preview_original = self.original_image.resize(
            preview_size, Image.Resampling.LANCZOS
        )
        self._preview_layers_cache = None
        self._preview_pixelated_cache.clear()

    def invalidate_preview_layers(self) -> None:
        self._preview_layers_cache = None

    def _preview_layers(self) -> tuple[Image.Image, Image.Image]:
        assert self.preview_original is not None
        assert self.mosaic_mask is not None and self.fill_layer is not None
        if self._preview_layers_cache is None:
            size = self.preview_original.size
            self._preview_layers_cache = (
                self.mosaic_mask.resize(size, Image.Resampling.NEAREST),
                self.fill_layer.resize(size, Image.Resampling.NEAREST),
            )
        return self._preview_layers_cache

    def _pixelated_preview(self, block_size: int) -> Image.Image:
        assert self.preview_original is not None
        cached = self._preview_pixelated_cache.get(block_size)
        if cached is None:
            cached = self._pixelate(self.preview_original.convert("RGB"), block_size)
            self._preview_pixelated_cache = {block_size: cached}
        return cached

    def begin_stroke(
        self,
        tool: ToolType,
        brush_size: int,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if not self.is_loaded or tool is ToolType.PAN:
            return
        self._stroke_mask = Image.new("L", self.size, 0)
        self._stroke_tool = tool
        self._stroke_color = color
        self._stroke_size = max(1, int(brush_size))
        self._stroke_bbox = None
        self._last_point = None

    def add_stroke_point(self, x: float, y: float) -> None:
        if self._stroke_mask is None:
            return
        width, height = self.size
        x = min(max(x, 0.0), width - 1.0)
        y = min(max(y, 0.0), height - 1.0)
        radius = self._stroke_size / 2
        draw = ImageDraw.Draw(self._stroke_mask)

        if self._last_point is None:
            points = [(x, y)]
        else:
            last_x, last_y = self._last_point
            distance = math.hypot(x - last_x, y - last_y)
            step = max(1.0, self._stroke_size / 4)
            count = max(1, math.ceil(distance / step))
            points = [
                (
                    last_x + (x - last_x) * index / count,
                    last_y + (y - last_y) * index / count,
                )
                for index in range(1, count + 1)
            ]

        for point_x, point_y in points:
            draw.ellipse(
                (
                    point_x - radius,
                    point_y - radius,
                    point_x + radius,
                    point_y + radius,
                ),
                fill=255,
            )
        self._last_point = (x, y)
        self._include_stroke_bounds(x, y, radius)

    def _include_stroke_bounds(self, x: float, y: float, radius: float) -> None:
        width, height = self.size
        box = (
            max(0, math.floor(x - radius - 1)),
            max(0, math.floor(y - radius - 1)),
            min(width, math.ceil(x + radius + 1)),
            min(height, math.ceil(y + radius + 1)),
        )
        if self._stroke_bbox is None:
            self._stroke_bbox = box
            return
        old = self._stroke_bbox
        self._stroke_bbox = (
            min(old[0], box[0]),
            min(old[1], box[1]),
            max(old[2], box[2]),
            max(old[3], box[3]),
        )

    def commit_stroke(self) -> LayerPatch | None:
        if self._stroke_mask is None or self._stroke_bbox is None:
            self.cancel_stroke()
            return None
        assert self.mosaic_mask is not None and self.fill_layer is not None
        bbox = self._stroke_bbox
        before_mosaic = self.mosaic_mask.crop(bbox)
        before_fill = self.fill_layer.crop(bbox)
        local_stroke = self._stroke_mask.crop(bbox)

        if self._stroke_tool is ToolType.MOSAIC:
            current = self.mosaic_mask.crop(bbox)
            self.mosaic_mask.paste(ImageChops.lighter(current, local_stroke), bbox[:2])
        elif self._stroke_tool is ToolType.FILL:
            color_layer = Image.new("RGBA", local_stroke.size, (*self._stroke_color, 255))
            self.fill_layer.paste(color_layer, bbox[:2], local_stroke)
        elif self._stroke_tool is ToolType.ERASER:
            self.mosaic_mask.paste(0, bbox, local_stroke)
            transparent = Image.new("RGBA", local_stroke.size, (0, 0, 0, 0))
            self.fill_layer.paste(transparent, bbox[:2], local_stroke)

        patch = LayerPatch(
            bbox=bbox,
            before_mosaic=before_mosaic,
            after_mosaic=self.mosaic_mask.crop(bbox),
            before_fill=before_fill,
            after_fill=self.fill_layer.crop(bbox),
        )
        self.invalidate_preview_layers()
        self.cancel_stroke()
        return patch

    def cancel_stroke(self) -> None:
        self._stroke_mask = None
        self._stroke_tool = None
        self._stroke_bbox = None
        self._last_point = None

    def clear_layers(self) -> LayerPatch | None:
        if not self.is_loaded:
            return None
        assert self.mosaic_mask is not None and self.fill_layer is not None
        if self.mosaic_mask.getbbox() is None and self.fill_layer.getchannel("A").getbbox() is None:
            return None
        bbox = (0, 0, *self.size)
        patch = LayerPatch(
            bbox=bbox,
            before_mosaic=self.mosaic_mask.copy(),
            after_mosaic=Image.new("L", self.size, 0),
            before_fill=self.fill_layer.copy(),
            after_fill=Image.new("RGBA", self.size, (0, 0, 0, 0)),
        )
        patch.apply(self, after=True)
        return patch

    def apply_mosaic_mask(self, mask: Image.Image) -> LayerPatch | None:
        if not self.is_loaded:
            return None
        assert self.mosaic_mask is not None and self.fill_layer is not None
        incoming = mask.convert("L")
        if incoming.size != self.size:
            raise ValueError("自動検出マスクのサイズが画像と一致しません。")
        bbox = incoming.getbbox()
        if bbox is None:
            return None
        before_mosaic = self.mosaic_mask.crop(bbox)
        before_fill = self.fill_layer.crop(bbox)
        combined = ImageChops.lighter(before_mosaic, incoming.crop(bbox))
        self.mosaic_mask.paste(combined, bbox[:2])
        patch = LayerPatch(
            bbox=bbox,
            before_mosaic=before_mosaic,
            after_mosaic=self.mosaic_mask.crop(bbox),
            before_fill=before_fill,
            after_fill=self.fill_layer.crop(bbox),
        )
        self.invalidate_preview_layers()
        return patch

    def render_preview(self, block_size: int, dilation_px: int = 0) -> Image.Image:
        if self.preview_original is None:
            raise RuntimeError("画像が読み込まれていません。")
        size = self.preview_original.size
        mosaic_mask, fill_layer = self._preview_layers()
        if self._stroke_mask is not None and self._stroke_tool is not None:
            stroke = self._stroke_mask.resize(size, Image.Resampling.NEAREST)
            mosaic_mask, fill_layer = self._apply_temporary_stroke(
                mosaic_mask, fill_layer, stroke
            )
        scaled_block = max(1, round(block_size * self.preview_scale))
        scaled_dilation = (
            max(1, round(dilation_px * self.preview_scale)) if dilation_px > 0 else 0
        )
        return self._compose(
            self.preview_original,
            mosaic_mask,
            fill_layer,
            scaled_block,
            scaled_dilation,
            pixelated=self._pixelated_preview(scaled_block),
        )

    def render_full(self, block_size: int, dilation_px: int = 0) -> Image.Image:
        if self.original_image is None:
            raise RuntimeError("画像が読み込まれていません。")
        assert self.mosaic_mask is not None and self.fill_layer is not None
        return self._compose(
            self.original_image,
            self.mosaic_mask,
            self.fill_layer,
            max(1, int(block_size)),
            max(0, int(dilation_px)),
        )

    def _apply_temporary_stroke(
        self, mosaic_mask: Image.Image, fill_layer: Image.Image, stroke: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if self._stroke_tool is ToolType.MOSAIC:
            mosaic_mask = ImageChops.lighter(mosaic_mask, stroke)
        elif self._stroke_tool is ToolType.FILL:
            color = Image.new("RGBA", fill_layer.size, (*self._stroke_color, 255))
            fill_layer = Image.composite(color, fill_layer, stroke)
        elif self._stroke_tool is ToolType.ERASER:
            inverse = ImageChops.invert(stroke)
            mosaic_mask = ImageChops.multiply(mosaic_mask, inverse)
            alpha = ImageChops.multiply(fill_layer.getchannel("A"), inverse)
            fill_layer = fill_layer.copy()
            fill_layer.putalpha(alpha)
        return mosaic_mask, fill_layer

    @classmethod
    def _compose(
        cls,
        original: Image.Image,
        mosaic_mask: Image.Image,
        fill_layer: Image.Image,
        block_size: int,
        dilation_px: int,
        *,
        pixelated: Image.Image | None = None,
    ) -> Image.Image:
        source = original.convert("RGBA")
        original_rgb = source.convert("RGB")
        if pixelated is None:
            pixelated = cls._pixelate(original_rgb, block_size)
        effective_mask = mosaic_mask
        if dilation_px > 0 and effective_mask.getbbox() is not None:
            effective_mask = cls._dilate_mask_by_pixels(effective_mask, dilation_px)
        result_rgb = Image.composite(pixelated, original_rgb, effective_mask)
        fill_rgb = fill_layer.convert("RGB")
        result_rgb = Image.composite(fill_rgb, result_rgb, fill_layer.getchannel("A"))
        result = result_rgb.convert("RGBA")
        result.putalpha(source.getchannel("A"))
        result.info.clear()
        return result

    @staticmethod
    def _dilate_mask_by_pixels(mask: Image.Image, radius: int) -> Image.Image:
        pixels = max(0, int(radius))
        if pixels <= 0 or mask.getbbox() is None:
            return mask
        return mask.filter(ImageFilter.MaxFilter(pixels * 2 + 1))

    @staticmethod
    def _dilate_mask_by_block(mask: Image.Image, block_size: int) -> Image.Image:
        """Expand occupied block cells by one cell without a huge max-filter kernel."""
        block = max(1, int(block_size))
        width, height = mask.size
        padded_width = math.ceil(width / block) * block
        padded_height = math.ceil(height / block) * block
        padded = Image.new("L", (padded_width, padded_height), 0)
        padded.paste(mask, (0, 0))
        grid_size = (padded_width // block, padded_height // block)
        occupied = padded.resize(grid_size, Image.Resampling.BOX).point(
            lambda value: 255 if value else 0
        )
        expanded = occupied.filter(ImageFilter.MaxFilter(3))
        return expanded.resize(padded.size, Image.Resampling.NEAREST).crop(
            (0, 0, width, height)
        )

    @staticmethod
    def _pixelate(image: Image.Image, block_size: int) -> Image.Image:
        block = max(1, int(block_size))
        width, height = image.size
        padded_width = math.ceil(width / block) * block
        padded_height = math.ceil(height / block) * block
        if (padded_width, padded_height) == image.size:
            padded = image
        else:
            padded = Image.new("RGB", (padded_width, padded_height))
            padded.paste(image, (0, 0))
            if padded_width > width:
                edge = image.crop((width - 1, 0, width, height)).resize(
                    (padded_width - width, height)
                )
                padded.paste(edge, (width, 0))
            if padded_height > height:
                edge = padded.crop((0, height - 1, padded_width, height)).resize(
                    (padded_width, padded_height - height)
                )
                padded.paste(edge, (0, height))
        small = padded.resize(
            (padded_width // block, padded_height // block), Image.Resampling.BOX
        )
        pixelated = small.resize(padded.size, Image.Resampling.NEAREST)
        return pixelated.crop((0, 0, width, height))

    def encode_png(self, block_size: int, dilation_px: int = 0) -> bytes:
        final_image = self.render_full(block_size, dilation_px)
        final_image.info.clear()
        buffer = io.BytesIO()
        final_image.save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=9,
            pnginfo=None,
        )
        return buffer.getvalue()

    def save_encoded_png(self, destination: str | os.PathLike[str], data: bytes) -> None:
        if self.source_path is None:
            raise RuntimeError("画像が読み込まれていません。")
        target = Path(destination)
        if target.suffix.lower() != ".png":
            raise ValueError("保存形式はPNGのみです。")
        if os.path.normcase(os.path.abspath(target)) == os.path.normcase(
            os.path.abspath(self.source_path)
        ):
            raise OriginalOverwriteError("元画像は上書きできません。")

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
