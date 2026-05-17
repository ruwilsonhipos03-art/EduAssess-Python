import json
import os
import sys
import tempfile
import traceback
from typing import Dict, Tuple

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import CheckBubbles

try:
    from pyzbar.pyzbar import decode as decode_qr
except Exception:
    decode_qr = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]  # Top-Left
    rect[2] = pts[np.argmax(sums)]  # Bottom-Right
    diffs = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diffs)]  # Top-Right
    rect[3] = pts[np.argmax(diffs)]  # Bottom-Left
    return rect


def warp_sheet_to_landscape(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    landscape_size = (1553, 1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    warped = cv2.resize(image, landscape_size)
    debug_meta = {"method": "fallback_scale", "page_corners": None}

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            rect = order_points(quad)
            
            dst = np.array([
                [0, 0],
                [landscape_size[0] - 1, 0],
                [landscape_size[0] - 1, landscape_size[1] - 1],
                [0, landscape_size[1] - 1]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, matrix, landscape_size)
            debug_meta = {
                "method": "largest_page_contour",
                "page_corners": rect.tolist(),
            }
            break
    return warped, debug_meta


def resolve_output_dir() -> str:
    configured = (os.getenv("OMR_OUTPUT_DIR") or "").strip()
    if configured:
        return os.path.abspath(configured)
    return os.path.abspath(os.path.join(CURRENT_DIR, "output"))


def _public_path(path: str, output_dir: str) -> str:
    public_prefix = (os.getenv("OMR_PUBLIC_DEBUG_PREFIX") or "omr_processed").strip("/")
    try:
        rel = os.path.relpath(path, output_dir)
    except ValueError:
        rel = os.path.basename(path)
    return f"{public_prefix}/{rel.replace(os.sep, '/')}"


def _decode_qr_with_pyzbar(image: np.ndarray) -> str:
    if not decode_qr:
        return ""
    try:
        decoded = decode_qr(image)
    except Exception:
        return ""
    if not decoded:
        return ""
    return decoded[0].data.decode("utf-8", errors="ignore").strip()


def _decode_qr_with_opencv(image: np.ndarray) -> str:
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(image)
    except Exception:
        return ""
    return (data or "").strip()


def _qr_variants(image: np.ndarray) -> list[np.ndarray]:
    variants = []
    base = image
    if len(base.shape) == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    variants.append(base)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    for scale in (2.0, 3.0):
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(resized)
        variants.append(cv2.copyMakeBorder(resized, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255))

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants.append(thresh)
    variants.append(cv2.copyMakeBorder(thresh, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255))
    return variants


def _qr_candidate_regions(image: np.ndarray) -> list[np.ndarray]:
    h, w = image.shape[:2]
    crops = [image]

    # Printed sheets place the QR at the page's top-left. Include other corners
    # so a rotated camera capture can still decode before/after normalization.
    crop_specs = [
        (0.00, 0.00, 0.42, 0.42),
        (0.00, 0.58, 0.42, 1.00),
        (0.58, 0.00, 1.00, 0.42),
        (0.58, 0.58, 1.00, 1.00),
        (0.00, 0.00, 0.55, 0.55),
    ]

    for y1, x1, y2, x2 in crop_specs:
        crop = image[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]
        if crop.size:
            crops.append(crop)

    return crops


def decode_sheet_qr(*images: np.ndarray) -> str:
    for image in images:
        if image is None:
            continue

        rotations = [
            image,
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(image, cv2.ROTATE_180),
        ]

        for rotated in rotations:
            for region in _qr_candidate_regions(rotated):
                for variant in _qr_variants(region):
                    data = _decode_qr_with_pyzbar(variant) or _decode_qr_with_opencv(variant)
                    if data:
                        return data

    return "UNKNOWN"


def process_document_result(image_path: str) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "IMAGE_READ_ERROR"}

    output_dir = resolve_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    root_name = os.path.splitext(filename)[0]

    # Step 1: Flatten Perspective Shape Map
    warped_landscape, warp_meta = warp_sheet_to_landscape(image)

    # Step 2: Rotate to Portrait Orientation
    portrait_image = cv2.rotate(warped_landscape, cv2.ROTATE_90_CLOCKWISE)

    original_path = os.path.join(output_dir, f"debug_{root_name}_00_original.jpg")
    perspective_path = os.path.join(output_dir, f"debug_{root_name}_02_perspective_warp.jpg")
    cv2.imwrite(original_path, image)
    cv2.imwrite(perspective_path, portrait_image)

    # Step 3: QR Parsing
    qr_data = decode_sheet_qr(image, warped_landscape, portrait_image)

    # Step 4: System Threshold Workspace
    gray = cv2.cvtColor(portrait_image, cv2.COLOR_BGR2GRAY)
    thresh_full = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    # Step 5: Extract Outer Container Box Contour Coordinates Dynamically
    contours, _ = cv2.findContours(thresh_full.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    bx, by, bw, bh = 0, 0, portrait_image.shape[1], portrait_image.shape[0]
    green_rect_visual = portrait_image.copy()

    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        if w_box > (portrait_image.shape[1] * 0.4) and h_box > (portrait_image.shape[0] * 0.4):
            bx, by, bw, bh = x, y, w_box, h_box
            cv2.rectangle(green_rect_visual, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            break

    green_rect_path = os.path.join(output_dir, f"debug_{root_name}_01_green_rectangles.jpg")
    cv2.imwrite(green_rect_path, green_rect_visual)

    # Isolate sub-grid image workspace (Image 03 Context)
    bubble_box_roi = thresh_full[by : by + bh, bx : bx + bw]
    before_check_path = os.path.join(output_dir, f"debug_{root_name}_03_before_check_bubbles.jpg")
    cv2.imwrite(before_check_path, bubble_box_roi)

    # Step 6: Process Matrix evaluation inside CheckBubbles using Image 03 directly
    bubble_result = CheckBubbles.analyze_bubbles(bubble_box_roi, filename)

    # Step 7: Final Visual Overlay Rendering (04_after_check_bubbles)
    after_bubbles_path = os.path.join(output_dir, f"debug_{root_name}_04_after_check_bubbles.jpg")
    final_debug_visual = portrait_image.copy()
    
    if "bubble_centers" in bubble_result:
        r_size = bubble_result.get("calculated_radius", 8)
        
        for center, is_filled, item_num_str in bubble_result["bubble_centers"]:
            # Recalculate local sub-grid points back onto portrait view coordinates
            cx = center[0] + bx
            cy = center[1] + by
            
            # Draw tracking circle for every identified option node
            cv2.circle(final_debug_visual, (cx, cy), r_size + 2, (0, 255, 0), 2)
            
            # Fill chosen bubble choices with a clean indicator marker
            if is_filled:
                cv2.circle(final_debug_visual, (cx, cy), int(r_size * 0.65), (0, 0, 255), -1)
                
    cv2.imwrite(after_bubbles_path, final_debug_visual)

    debug_images = {
        "original": _public_path(original_path, output_dir),
        "green_rectangles": _public_path(green_rect_path, output_dir),
        "perspective_warp": _public_path(perspective_path, output_dir),
        "before_check_bubbles": _public_path(before_check_path, output_dir),
        "after_check_bubbles": _public_path(after_bubbles_path, output_dir),
    }

    return {
        "file": filename,
        "sheet_id": qr_data,
        "answers": bubble_result.get("answers", {}),
        "warp_debug_path": debug_images["perspective_warp"],
        "debug": debug_images["after_check_bubbles"],
        "debug_images": debug_images,
        "warp_debug": warp_meta,
    }


def detect_bubble_grid(img, filename: str = "") -> Dict:
    if isinstance(img, (str, os.PathLike)):
        return process_document_result(os.fspath(img))

    temp_path = None
    try:
        ext = os.path.splitext(filename or "capture.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_path = tmp.name
        cv2.imwrite(temp_path, img)
        return process_document_result(temp_path)
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "NO_IMAGE_PATH"}))
        return
    print(json.dumps(process_document_result(sys.argv[1])))


if __name__ == "__main__":
    main()
