"""Inference logic for the demo, kept free of Streamlit so it can be tested directly."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install at runtime

from ultralytics import YOLO

HERE = Path(__file__).parent
# A1-A4 and B1-B3 were pickled with the top-level modules `cbam` / `attention`
# (copied verbatim from code-v2/code/ktln). Put them first on sys.path and drop
# any same-named third-party package already imported (pip "attention" exists).
sys.path.insert(0, str(HERE))
for _name in ("cbam", "attention"):
    _mod = sys.modules.get(_name)
    if _mod is not None and Path(getattr(_mod, "__file__", "") or "").parent != HERE:
        del sys.modules[_name]

MODELS_DIR = HERE / "trong_so"  # <Mã>/s<seed>/best.pt + thresholds.json
# Code -> display name (results-exp2-10112026/"ID và tên mô hình.md").
MODELS = {
    "A0": "YOLO11n (baseline)",
    "A1": "YOLO11n + CBAM-Backbone",
    "A2": "YOLO11n + CBAM-Neck",
    "A3": "YOLO11n + CBAM-PreDetect",
    "A4": "YOLO11n + CBAM-Backbone+PreDetect",
    "B1": "YOLO11n + SimAM",
    "B2": "YOLO11n + EMA",
    "B3": "YOLO11n + CBAM-PreDetect (khởi tạo mặc định)",
    "D1": "YOLO11s",
    "D2": "YOLOv8n",
}
# Seed with the highest val F1 at tau* among seeds 42-46 (f1_val_at_tau_star in each
# run's thresholds.json); the test set is not used for this choice.
SEEDS = {
    "A0": 43, "A1": 43, "A2": 46, "A3": 43, "A4": 45,
    "B1": 45, "B2": 46, "B3": 46, "D1": 43, "D2": 45,
}
DEFAULT_MODEL = "A0"
WEIGHTS = MODELS_DIR / DEFAULT_MODEL / f"s{SEEDS[DEFAULT_MODEL]}" / "best.pt"
TAU_STAR = 0.12  # A0 s43, chosen on val; per-model values come from tau_star()
IMGSZ = 1024
IOU = 0.7
MIN_CONF = 0.01  # lowest value the threshold slider can reach
MAX_SIDE = 8000  # refuse absurdly large images instead of exhausting memory


class InvalidImage(ValueError):
    """The uploaded bytes are not an image the system can process."""


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]


@dataclass
class RawResult:
    detections: list[Detection]
    width: int
    height: int
    decode_ms: float
    infer_ms: float


def weights_path(code: str) -> Path:
    return MODELS_DIR / code / f"s{SEEDS[code]}" / "best.pt"


def available_models() -> list[str]:
    return [c for c in MODELS if weights_path(c).is_file()]


def tau_star(code: str) -> float:
    """Val-selected threshold saved next to the weights at training time."""
    f = MODELS_DIR / code / f"s{SEEDS[code]}" / "thresholds.json"
    return float(json.loads(f.read_text(encoding="utf-8"))["tau_star"])


def load_model(path: Path = WEIGHTS) -> YOLO:
    import cbam
    cbam.register()  # also registers SimAM/EMA; needed to unpickle A1-A4, B1-B3
    return YOLO(str(path))


def decode_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        raise InvalidImage("Tệp không phải ảnh hợp lệ hoặc đã bị hỏng.") from e
    if max(img.size) > MAX_SIDE:
        raise InvalidImage(f"Ảnh quá lớn (cạnh dài tối đa {MAX_SIDE} px).")
    return ImageOps.exif_transpose(img).convert("RGB")


def detect(model: YOLO, data: bytes) -> RawResult:
    """Run the model once at the lowest confidence; the threshold is applied afterwards."""
    t0 = time.perf_counter()
    img = decode_image(data)
    t1 = time.perf_counter()
    r = model.predict(np.asarray(img)[:, :, ::-1], imgsz=IMGSZ, conf=MIN_CONF, iou=IOU,
                      max_det=300, verbose=False)[0]
    t2 = time.perf_counter()
    dets = [Detection(model.names[int(c)], float(p), tuple(map(float, b)))
            for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return RawResult(dets, img.width, img.height, (t1 - t0) * 1000, (t2 - t1) * 1000)


def apply_threshold(raw: RawResult, tau: float) -> list[Detection]:
    return sorted((d for d in raw.detections if d.conf >= tau), key=lambda d: -d.conf)


def image_score(raw: RawResult) -> float:
    return max((d.conf for d in raw.detections), default=0.0)


def annotate(data: bytes, dets: list[Detection]) -> Image.Image:
    img = decode_image(data)
    draw = ImageDraw.Draw(img)
    scale = max(img.size) / 900
    font = ImageFont.load_default(size=max(12, int(18 * scale)))
    width = max(2, int(3 * scale))
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        draw.rectangle((x1, y1, x2, y2), outline="#E5392E", width=width)
        label = f"{d.cls} {d.conf:.2f}"
        box = draw.textbbox((x1, y1), label, font=font)
        h = box[3] - box[1] + 6
        top = y1 - h if y1 >= h else y1
        draw.rectangle((x1, top, x1 + box[2] - box[0] + 8, top + h), fill="#E5392E")
        draw.text((x1 + 4, top + 1), label, fill="white", font=font)
    return img
