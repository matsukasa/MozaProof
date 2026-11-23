from __future__ import annotations

import io
import math
import os
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .history import LayerPatch
from .settings import PREVIEW_MAX_EDGE
from .tools import ToolType


class ImageLoadError(ValueError):
    pass


class OriginalOverwriteError(ValueError):
    pass


def _read_png_metadata(source, *, grayscale: bool) -> tuple[bytes, tuple[str, ...]]:
    # Copy only chunks that remain meaningful when the pixels are saved as RGBA.
    supported = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"pHYs", b"gAMA", b"cHRM", b"sRGB", b"cICP", b"mDCV", b"cLLI"}
    if not grayscale:
        supported.add(b"iCCP")
    structural = {b"IHDR", b"PLTE", b"IDAT", b"tRNS"}
    metadata = bytearray()
    unsupported = set()
    file_size = source.seek(0, os.SEEK_END)
    source.seek(8)
    while source.tell() < file_size:
        header = source.read(8)
        if len(header) != 8:
            break
        length = int.from_bytes(header[:4], "big")
        kind = header[4:]
        if kind == b"IEND":
            break
        if length + 4 > file_size - source.tell():
            raise ValueError("PNGチャンクの長さが不正です。")
        if kind in supported:
            data = source.read(length)
            crc = source.read(4)
            if zlib.crc32(data, zlib.crc32(kind)) == int.from_bytes(crc, "big"):
                metadata.extend(header)
                metadata.extend(data)
                metadata.extend(crc)
            else:
                unsupported.add(kind)
        else:
            source.seek(length + 4, os.SEEK_CUR)
            if kind not in structural:
                unsupported.add(kind)
    return bytes(metadata), tuple(k.decode("ascii", "replace") for k in sorted(unsupported))


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
        self._png_metadata = b""
        self.unsupported_png_chunks: tuple[str, ...] = ()
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
            with source.open("rb") as stream, Image.open(stream) as opened:
                if opened.format != "PNG":
                    raise ImageLoadError("PNGのみ対応しています。")
                opened.load()
                original = opened.convert("RGBA")
                metadata, unsupported = _read_png_metadata(
                    stream, grayscale=opened.mode not in ("RGB", "RGBA", "P", "PA")
                )
        except ImageLoadError:
            raise
        except (OSError, ValueError) as exc:
            raise ImageLoadError(f"PNG画像を読み込めませんでした: {exc}") from exc

        self.source_path = source.resolve()
        self.original_image = original
        self._png_metadata = metadata
        self.unsupported_png_chunks = unsupported
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
            # BILINEAR, not NEAREST: at preview scale a 1px mask stroke can
            # fall between NEAREST sample points and vanish from the
            # preview while still applying to the saved image. A graded
            # BILINEAR edge keeps every stroke visible.
            self._preview_layers_cache = (
                self.mosaic_mask.resize(size, Image.Resampling.BILINEAR),
                self.fill_layer.resize(size, Image.Resampling.BILINEAR),
            )
        return self._preview_layers_cache

    def _pixelated_preview(self, block_size: int) -> Image.Image:
        assert self.preview_original is not None
        cached = self._preview_pixelated_cache.get(block_size)
        if cached is None:
            cached = self._pixelate(self.preview_original, block_size)
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
            stroke = self._stroke_mask.resize(size, Image.Resampling.BILINEAR)
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
        result = original.copy()
        effective_mask = mosaic_mask
        if dilation_px > 0:
            effective_mask = cls._dilate_mask_by_pixels(effective_mask, dilation_px)
        bounds = effective_mask.getbbox()
        if bounds and block_size > 1:
            # Align the crop to the same grid as a full-image render.
            x1, y1, x2, y2 = bounds
            width, height = original.size
            box = (
                x1 - x1 % block_size, y1 - y1 % block_size,
                min(width, x2 + (-x2) % block_size),
                min(height, y2 + (-y2) % block_size),
            )
            if pixelated is None:
                region = original if box == (0, 0, width, height) else original.crop(box)
                mosaic = cls._pixelate(region, block_size)
            else:
                mosaic = pixelated.crop(box)
            result.paste(mosaic, box, effective_mask.crop(box))
        fill_bounds = fill_layer.getbbox()
        if fill_bounds:
            fill = fill_layer.crop(fill_bounds)
            result.paste(fill, fill_bounds, fill.getchannel("A"))
        if bounds or fill_bounds:
            result.putalpha(original.getchannel("A"))
        result.info.clear()
        return result

    @staticmethod
    def _dilate_mask_by_pixels(mask: Image.Image, radius: int) -> Image.Image:
        pixels = max(0, int(radius))
        if pixels <= 0 or mask.getbbox() is None:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pixels * 2 + 1,) * 2)
        return Image.fromarray(cv2.dilate(np.asarray(mask), kernel))

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
        if block == 1:
            return image
        pixels = np.asarray(image)
        pad_right, pad_bottom = (-width) % block, (-height) % block
        if pad_right or pad_bottom:
            pixels = cv2.copyMakeBorder(pixels, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
        padded_height, padded_width = pixels.shape[:2]
        cells = cv2.resize(
            pixels, (padded_width // block, padded_height // block),
            interpolation=cv2.INTER_AREA,
        )
        expanded = cv2.resize(
            cells, (padded_width, padded_height), interpolation=cv2.INTER_NEAREST
        )
        result = Image.fromarray(expanded)
        return result.crop((0, 0, width, height)) if pad_right or pad_bottom else result

    def encode_png(
        self, block_size: int, dilation_px: int = 0, *, compress_level: int = 1,
        keep_metadata: bool = False,
    ) -> bytes:
        if not isinstance(compress_level, int) or not 0 <= compress_level <= 9:
            raise ValueError("PNG圧縮レベルは0〜9で指定してください。")
        final_image = self.render_full(block_size, dilation_px)
        final_image.info.clear()
        buffer = io.BytesIO()
        final_image.save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=compress_level,
            pnginfo=None,
        )
        if keep_metadata and self._png_metadata:
            # Insert the original ancillary chunks after the fixed signature and IHDR.
            with buffer.getbuffer() as png:
                return b"".join((png[:33], self._png_metadata, png[33:]))
        return buffer.getvalue()

    def save_encoded_png(self, destination: str | os.PathLike[str], data: bytes) -> None:
        if self.source_path is None:
            raise RuntimeError("画像が読み込まれていません。")
        target = Path(destination)
        if target.suffix.lower() != ".png":
            raise ValueError("保存形式はPNGのみです。")
        if os.path.normcase(target.resolve()) == os.path.normcase(
            Path(self.source_path).resolve()
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
