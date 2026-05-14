import os
import tempfile
from typing import Callable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from CheckBubbles import detect_bubble_grid as detect_bubbles
from CheckExam import detect_bubble_grid as detect_exam


app = FastAPI(title="EduAssess OMR Service", version="1.0.0")


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = _order_points(points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 10 or max_height < 10:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def _scan_preprocess(image: np.ndarray) -> np.ndarray:
    """Try to flatten and crop a sheet-like contour, fallback to original image."""
    original = image
    height = image.shape[0]
    scale = height / 500.0 if height > 0 else 1.0
    resized = image if height <= 500 else cv2.resize(
        image, (int(image.shape[1] / scale), 500), interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    page = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx.reshape(4, 2).astype("float32")
            break

    if page is None:
        return original

    warped = _four_point_transform(original, page * scale)

    # Optional "scanned paper" black/white look if enabled.
    if str(os.getenv("OMR_APPLY_SCAN_THRESHOLD", "0")).strip() in {"1", "true", "True"}:
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(
            warped_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            10,
        )
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    return warped


def _require_api_key(x_api_key: Optional[str]) -> None:
    expected = (os.getenv("OMR_API_KEY") or "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _scan_uploaded_file(
    upload: UploadFile,
    detector: Callable[[cv2.typing.MatLike, str], dict],
) -> dict:
    filename = upload.filename or "upload.jpg"
    ext = os.path.splitext(filename)[1] or ".jpg"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_path = tmp.name
            tmp.write(upload.file.read())

        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        processed = _scan_preprocess(img)
        result = detector(processed, os.path.basename(temp_path))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        upload.file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/scan/exam")
def scan_exam(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    _require_api_key(x_api_key)
    exam_result = _scan_uploaded_file(file, detect_exam)

    # Run bubble-only pass automatically after exam detection.
    # Reconstruct image from exam output path if available, otherwise fail softly.
    processed_path = exam_result.get("processed_path")
    if processed_path and os.path.exists(processed_path):
        bubble_img = cv2.imread(processed_path)
        if bubble_img is not None:
            bubble_result = detect_bubbles(
                bubble_img,
                os.path.basename(processed_path),
            )
        else:
            bubble_result = {"error": "Cannot read processed image for bubble pass"}
    else:
        bubble_result = {"error": "No processed image path returned by exam pass"}

    return {
        "exam": exam_result,
        "check_bubbles": bubble_result,
    }


@app.post("/api/entrance/omr/check")
def scan_exam_laravel_alias(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    """Compatibility alias for Laravel/frontend route expectations."""
    return scan_exam(file=file, x_api_key=x_api_key)
