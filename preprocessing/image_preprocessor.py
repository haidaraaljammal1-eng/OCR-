"""Conservative image preprocessing variants for OCR retry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessVariant:
    name: str
    path: Path
    description: str


def _load_bgr(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _save_variant(bgr: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path)
    return out_path


def _deskew(bgr: np.ndarray, max_angle: float = 8.0) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th < 200))
    if coords.size < 100:
        return bgr, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return bgr, 0.0
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), angle


def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _mild_sharpen(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
    sharp = cv2.filter2D(bgr, -1, kernel)
    return cv2.addWeighted(bgr, 0.65, sharp, 0.35, 0)


def _adaptive_thresh_variant(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _mrz_crop(bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0 = int(h * 0.72)
    crop = bgr[y0:h, 0:w]
    if crop.size == 0:
        crop = bgr[int(h * 0.6) : h, 0:w]
    crop = _clahe_gray(crop)
    crop = _mild_sharpen(crop)
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def _licence_text_crop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.22), int(h * 0.88)
    x0, x1 = int(w * 0.08), int(w * 0.92)
    crop = bgr[y0:y1, x0:x1]
    crop = _clahe_gray(crop)
    crop = _mild_sharpen(crop)
    return crop


def generate_variants(
    source_image: Path,
    out_root: Path,
    doc_kind: str,
) -> list[PreprocessVariant]:
    """Build conservative variants; originals are never modified."""
    bgr = _load_bgr(source_image)
    stem = source_image.stem
    dest_dir = out_root / stem
    variants: list[PreprocessVariant] = []

    deskewed, angle = _deskew(bgr)
    if abs(angle) > 0.3:
        p = _save_variant(deskewed, dest_dir / "deskew.png")
        variants.append(PreprocessVariant("deskew", p, f"Deskew {angle:.2f}°"))

    p = _save_variant(_clahe_gray(bgr), dest_dir / "grayscale_clahe.png")
    variants.append(PreprocessVariant("grayscale_clahe", p, "Grayscale + mild CLAHE"))

    p = _save_variant(_mild_sharpen(bgr), dest_dir / "mild_sharpen.png")
    variants.append(PreprocessVariant("mild_sharpen", p, "Mild sharpen"))

    combo = _mild_sharpen(_clahe_gray(deskewed))
    p = _save_variant(combo, dest_dir / "deskew_clahe_sharpen.png")
    variants.append(PreprocessVariant("deskew_clahe_sharpen", p, "Deskew + CLAHE + sharpen"))

    p = _save_variant(_adaptive_thresh_variant(bgr), dest_dir / "adaptive_threshold.png")
    variants.append(PreprocessVariant("adaptive_threshold", p, "Adaptive threshold (mild)"))

    if doc_kind == "passport":
        p = _save_variant(_mrz_crop(bgr), dest_dir / "mrz_crop.png")
        variants.append(PreprocessVariant("mrz_crop", p, "Bottom MRZ band, 2x, CLAHE"))

    if doc_kind == "licence":
        p = _save_variant(_licence_text_crop(bgr), dest_dir / "licence_fields_crop.png")
        variants.append(
            PreprocessVariant("licence_fields_crop", p, "Central text band for number/dates")
        )

    return variants
