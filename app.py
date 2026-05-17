import os
import tempfile
from typing import Callable, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from CheckBubbles import detect_bubble_grid as detect_bubbles
from CheckExam import detect_bubble_grid as detect_exam

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

app = FastAPI(title="EduAssess OMR Service", version="1.0.0")

def _require_api_key(x_api_key: Optional[str]) -> None:
    expected = (os.getenv("OMR_API_KEY") or os.getenv("OMR_API_BEARER_TOKEN") or "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _scan_uploaded_file_raw(upload: UploadFile, detector: Callable[[str], dict]) -> dict:
    filename = upload.filename or "upload.jpg"
    ext = os.path.splitext(filename)[1] or ".jpg"
    temp_path = None

    try:
        # Read the raw byte data streamed from Laravel first
        file_bytes = upload.file.read()
        if not file_bytes:
            raise ValueError("Received an empty file payload from Laravel.")

        # Reconstruct the image matrix directly from memory for verification
        nparr = np.frombuffer(file_bytes, np.uint8)
        debug_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- DEBUG DUMP ---
        if debug_img is not None:
            cv2.imwrite("laravel_received_dump.jpg", debug_img)
        else:
            raise ValueError("Uploaded data could not be decoded into a valid image matrix.")
        # ------------------

        # Safely write to disk and close the file completely before passing to detector
        # This resolves Windows file locking and un-flushed memory buffer issues.
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_path = tmp.name
            tmp.write(file_bytes)
            tmp.flush() # Ensure bytes are fully pushed to storage hardware

        # Verify the file actually exists and has content before executing detector
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise FileNotFoundError(f"Temporary file write failed or file is empty at: {temp_path}")

        # Now pass the path to your OMR scanner functions safely
        return detector(temp_path)

    except Exception as exc:
        # Capture and return the real underlying error trace back to Laravel
        raise HTTPException(status_code=500, detail=f"OMR Python Error: {str(exc)}")
    finally:
        upload.file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/health")
def health() -> dict:
    return {"ok": True}

@app.post("/scan/exam")
def scan_exam(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    return _scan_uploaded_file_raw(file, detect_exam)

@app.post("/scan/bubbles")
def scan_bubbles(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    return _scan_uploaded_file_raw(file, detect_exam)

@app.post("/api/entrance/omr/check")
def scan_exam_laravel_alias(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_exam(file=file, x_api_key=x_api_key)

@app.post("/api/omr/check-exam")
def scan_exam_test_alias(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_exam(file=file, x_api_key=x_api_key)

@app.post("/scan/term")
def scan_term(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_exam(file=file, x_api_key=x_api_key)

@app.post("/api/instructor/omr/check-term")
def scan_term_laravel_alias(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_term(file=file, x_api_key=x_api_key)

@app.post("/api/omr/check-term")
def scan_term_test_alias(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_term(file=file, x_api_key=x_api_key)

@app.post("/api/entrance/omr/check-bubbles")
def scan_bubbles_laravel_alias(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)) -> dict:
    return scan_bubbles(file=file, x_api_key=x_api_key)
