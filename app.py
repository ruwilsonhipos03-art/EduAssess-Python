import os
import tempfile
from typing import Callable, Optional

import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from CheckExam import detect_bubble_grid as detect_exam
from CheckTermExam import detect_bubble_grid as detect_term


app = FastAPI(title="EduAssess OMR Service", version="1.0.0")


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

        result = detector(img, os.path.basename(temp_path))
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
    return _scan_uploaded_file(file, detect_exam)


@app.post("/scan/term")
def scan_term(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    _require_api_key(x_api_key)
    return _scan_uploaded_file(file, detect_term)
