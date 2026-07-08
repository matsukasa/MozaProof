from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageChops, ImageDraw
from PySide6.QtCore import QObject, QThread, Signal, Slot


TARGET_CLASSES = frozenset(
    {
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
    }
)
HOTSCREEN_LABELS = (
    "FEMALE_FACE",
    "MALE_FACE",
    "FEMALE_GENITALIA_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "BUTTOCKS_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_BREAST_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_COVERED",
    "FEET_EXPOSED",
    "EYE",
)
DEFAULT_CONFIDENCE = 0.15


def resource_path(relative_path: str) -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / relative_path
    return Path(__file__).resolve().parents[1] / relative_path


@dataclass(slots=True)
class AutoCensorResult:
    mask: Image.Image
    detection_count: int
    segmented_count: int


class HotscreenDetector:
    input_size = 640
    iou_threshold = 0.45
    tile_threshold = 960
    tile_fraction = 0.62

    def __init__(self, model_dir: Path) -> None:
        model_path = model_dir / "hs-real-anime-y11n-640-fp32.onnx"
        if not model_path.is_file():
            raise FileNotFoundError("アニメ対応の自動検出モデルが見つかりません。")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def _preprocess(
        cls, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        height, width = image_bgr.shape[:2]
        scale = min(cls.input_size / width, cls.input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        left = (cls.input_size - resized_width) // 2
        top = (cls.input_size - resized_height) // 2
        padded = np.full((cls.input_size, cls.input_size, 3), 114, dtype=np.uint8)
        padded[top : top + resized_height, left : left + resized_width] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        )
        return tensor, scale, (left, top)

    @classmethod
    def _postprocess(
        cls,
        output: np.ndarray,
        image_shape: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
        confidence: float,
    ) -> list[dict]:
        prediction = np.squeeze(output)
        if prediction.ndim != 2:
            raise ValueError("自動検出モデルの出力形式が不正です。")
        if prediction.shape[0] == 4 + len(HOTSCREEN_LABELS):
            prediction = prediction.T
        if prediction.shape[1] != 4 + len(HOTSCREEN_LABELS):
            raise ValueError("自動検出モデルのクラス数が一致しません。")

        class_scores = prediction[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]
        keep = scores >= confidence
        if not np.any(keep):
            return []

        boxes = prediction[keep, :4].copy()
        scores = scores[keep]
        class_ids = class_ids[keep]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        left, top = pad
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top) / scale
        height, width = image_shape
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, width)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, height)

        boxes_xywh = [
            [
                float(x1),
                float(y1),
                float(max(0, x2 - x1)),
                float(max(0, y2 - y1)),
            ]
            for x1, y1, x2, y2 in boxes_xyxy
        ]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores.astype(float).tolist(),
            confidence,
            cls.iou_threshold,
        )
        if len(indices) == 0:
            return []

        detections = []
        for index in np.asarray(indices).reshape(-1):
            x, y, box_width, box_height = boxes_xywh[int(index)]
            if box_width < 1 or box_height < 1:
                continue
            detections.append(
                {
                    "class": HOTSCREEN_LABELS[int(class_ids[index])],
                    "score": float(scores[index]),
                    "box": [round(x), round(y), round(box_width), round(box_height)],
                }
            )
        return detections

    def _detect_single(
        self, image_bgr: np.ndarray, confidence: float = DEFAULT_CONFIDENCE
    ) -> list[dict]:
        tensor, scale, pad = self._preprocess(image_bgr)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._postprocess(
            output, image_bgr.shape[:2], scale, pad, confidence
        )

    @classmethod
    def _deduplicate(
        cls, detections: list[dict], confidence: float
    ) -> list[dict]:
        merged: list[dict] = []
        for label in HOTSCREEN_LABELS:
            candidates = [item for item in detections if item["class"] == label]
            if not candidates:
                continue
            indices = cv2.dnn.NMSBoxes(
                [item["box"] for item in candidates],
                [float(item["score"]) for item in candidates],
                confidence,
                cls.iou_threshold,
            )
            for index in np.asarray(indices).reshape(-1):
                merged.append(candidates[int(index)])
        return sorted(merged, key=lambda item: float(item["score"]), reverse=True)

    def detect(
        self, image_bgr: np.ndarray, confidence: float = DEFAULT_CONFIDENCE
    ) -> list[dict]:
        detections = self._detect_single(image_bgr, confidence)
        height, width = image_bgr.shape[:2]
        if max(width, height) <= self.tile_threshold:
            return detections

        tile_width = max(self.input_size, round(width * self.tile_fraction))
        tile_height = max(self.input_size, round(height * self.tile_fraction))
        tile_width = min(width, tile_width)
        tile_height = min(height, tile_height)
        starts = {
            (0, 0),
            (width - tile_width, 0),
            (0, height - tile_height),
            (width - tile_width, height - tile_height),
        }
        for left, top in starts:
            tile = image_bgr[top : top + tile_height, left : left + tile_width]
            for item in self._detect_single(tile, confidence):
                translated = dict(item)
                box = item["box"]
                translated["box"] = [
                    box[0] + left,
                    box[1] + top,
                    box[2],
                    box[3],
                ]
                detections.append(translated)
        return self._deduplicate(detections, confidence)


class MobileSamSegmenter:
    input_size = (684, 1024)
    target_size = 1024

    def __init__(self, model_dir: Path) -> None:
        encoder_path = model_dir / "mobile_sam_image_encoder.onnx"
        decoder_path = model_dir / "sam_mask_decoder_single.onnx"
        if not encoder_path.is_file() or not decoder_path.is_file():
            raise FileNotFoundError("MobileSAMモデルが見つかりません。")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            str(encoder_path), sess_options=options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), sess_options=options, providers=providers
        )
        self.encoder_input_name = self.encoder.get_inputs()[0].name

    def encode(self, image_bgr: np.ndarray) -> dict[str, np.ndarray | tuple[int, int]]:
        original_size = image_bgr.shape[:2]
        scale = min(
            self.input_size[1] / original_size[1],
            self.input_size[0] / original_size[0],
        )
        transform = np.array(
            [[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32
        )
        resized = cv2.warpAffine(
            image_bgr,
            transform[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32)
        embedding = self.encoder.run(
            None, {self.encoder_input_name: resized}
        )[0]
        return {
            "image_embedding": embedding,
            "transform": transform,
            "original_size": original_size,
        }

    def segment_box(
        self,
        encoded: dict[str, np.ndarray | tuple[int, int]],
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = box
        points = np.array(
            [[[x1, y1], [x2, y2], [0, 0]]], dtype=np.float32
        )
        homogeneous = np.concatenate(
            [points, np.ones((1, points.shape[1], 1), dtype=np.float32)], axis=2
        )
        transform = encoded["transform"]
        assert isinstance(transform, np.ndarray)
        transformed = np.matmul(homogeneous, transform.T)[:, :, :2].astype(
            np.float32
        )
        labels = np.array([[2, 3, -1]], dtype=np.float32)
        embedding = encoded["image_embedding"]
        assert isinstance(embedding, np.ndarray)
        masks, _, _ = self.decoder.run(
            None,
            {
                "image_embeddings": embedding,
                "point_coords": transformed,
                "point_labels": labels,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array(self.input_size, dtype=np.float32),
            },
        )
        original_size = encoded["original_size"]
        assert isinstance(original_size, tuple)
        inverse = np.linalg.inv(transform)
        restored = cv2.warpAffine(
            masks[0, 0],
            inverse[:2],
            (original_size[1], original_size[0]),
            flags=cv2.INTER_LINEAR,
        )
        mask = restored > 0
        constrained = np.zeros_like(mask)
        constrained[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return constrained


class AutoCensorEngine:
    def __init__(
        self,
        model_dir: Path | None = None,
        detector=None,
        segmenter_factory: Callable[[Path], MobileSamSegmenter] | None = None,
    ) -> None:
        self.model_dir = model_dir or resource_path("models")
        self.detector = detector or HotscreenDetector(self.model_dir)
        self.segmenter_factory = segmenter_factory or MobileSamSegmenter

    @staticmethod
    def _detected_box(
        box: list[int] | tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = (int(value) for value in box)
        image_width, image_height = image_size
        return (
            max(0, x),
            max(0, y),
            min(image_width, x + width),
            min(image_height, y + height),
        )

    @staticmethod
    def _ellipse_fallback(
        output: Image.Image, box: tuple[int, int, int, int]
    ) -> None:
        ImageDraw.Draw(output).ellipse(box, fill=255)

    def detect(
        self,
        image_path: str | os.PathLike[str],
        confidence: float = DEFAULT_CONFIDENCE,
        cancelled: Callable[[], bool] | None = None,
    ) -> AutoCensorResult:
        cancel_check = cancelled or (lambda: False)
        source = Path(image_path)
        with Image.open(source) as opened:
            image_size = opened.size
            image_rgb = opened.convert("RGB")
        output = Image.new("L", image_size, 0)
        image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
        detections = [
            item
            for item in self.detector.detect(image_bgr)
            if item.get("class") in TARGET_CLASSES
            and float(item.get("score", 0)) >= confidence
        ]
        if not detections or cancel_check():
            return AutoCensorResult(output, len(detections), 0)

        segmenter = self.segmenter_factory(self.model_dir)
        encoded = segmenter.encode(image_bgr)
        segmented_count = 0

        for detection in detections:
            if cancel_check():
                break
            box = self._detected_box(detection["box"], image_size)
            shape_mask = segmenter.segment_box(encoded, box)
            if shape_mask.any():
                shape_image = Image.fromarray(
                    np.where(shape_mask, 255, 0).astype(np.uint8), mode="L"
                )
                output = ImageChops.lighter(output, shape_image)
            else:
                self._ellipse_fallback(output, box)
            segmented_count += 1

        return AutoCensorResult(output, len(detections), segmented_count)


class AutoCensorWorker(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, generation: int, image_path: str) -> None:
        super().__init__()
        self.generation = generation
        self.image_path = image_path

    @Slot()
    def run(self) -> None:
        thread = QThread.currentThread()
        try:
            result = AutoCensorEngine().detect(
                self.image_path,
                cancelled=thread.isInterruptionRequested,
            )
            self.finished.emit(self.generation, result)
        except Exception as exc:  # Report model/runtime errors without crashing the UI.
            self.failed.emit(self.generation, str(exc))
