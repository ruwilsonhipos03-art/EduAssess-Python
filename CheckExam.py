import json
import os
import sys
import tempfile
import traceback

import cv2
import numpy as np

from CheckBubbles import analyze_bubbles

try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    InferenceHTTPClient = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


REQUIRED_CLASSES = ["top-left", "top-right", "bottom-right", "bottom-left"]
A4_SHORT = 2480
A4_LONG = 3508
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_MIN_POLYGON_AREA_RATIO = 0.08
DEFAULT_STRONG_CORNER_CONFIDENCE = 0.60


def _env_float(name, default):
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _min_corner_confidence(best_by_class):
    if not best_by_class:
        return 0.0
    vals = [float(v.get("confidence", 0.0)) for v in best_by_class.values()]
    return min(vals) if vals else 0.0


def resolve_output_dir(script_dir):
    configured = (os.getenv("OMR_OUTPUT_DIR") or "").strip()
    if configured:
        return os.path.abspath(configured)
    return os.path.abspath(
        os.path.join(script_dir, "omr","output")
    )


def select_best_predictions(predictions, threshold):
    best = {}
    for pred in predictions:
        cls = (pred.get("class") or "").strip().lower().replace("_", "-")
        conf = float(pred.get("confidence", 0))
        if cls not in REQUIRED_CLASSES or conf < threshold:
            continue
        if cls not in best or conf > float(best[cls].get("confidence", 0)):
            best[cls] = pred
    return best


def center_from_prediction(pred):
    return [float(pred["x"]), float(pred["y"])]


def order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _polygon_area(points):
    pts = np.array(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_point(pt, width, height):
    x = float(np.clip(pt[0], 0, max(0, width - 1)))
    y = float(np.clip(pt[1], 0, max(0, height - 1)))
    return [x, y]


def _recover_missing_corner(best_by_class, width, height):
    keys = set(best_by_class.keys())
    missing = [k for k in REQUIRED_CLASSES if k not in keys]
    if len(missing) != 1:
        return best_by_class, False

    m = missing[0]
    get = lambda k: np.array(center_from_prediction(best_by_class[k]), dtype=np.float32)

    try:
        if m == "top-left":
            inferred = get("top-right") + get("bottom-left") - get("bottom-right")
        elif m == "top-right":
            inferred = get("top-left") + get("bottom-right") - get("bottom-left")
        elif m == "bottom-right":
            inferred = get("top-right") + get("bottom-left") - get("top-left")
        else:  # bottom-left
            inferred = get("top-left") + get("bottom-right") - get("top-right")
    except Exception:
        return best_by_class, False

    clipped = _clip_point(inferred, width, height)
    out = dict(best_by_class)
    out[m] = {
        "class": m,
        "x": clipped[0],
        "y": clipped[1],
        "confidence": 0.0,
        "inferred": True,
    }
    return out, True


def validate_angle_and_get_size(ordered_pts):
    tl, tr, br, bl = ordered_pts
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    avg_w = (top_w + bottom_w) / 2.0
    avg_h = (left_h + right_h) / 2.0

    if avg_h <= 1 or avg_w <= 1:
        return (True, None, None)

    detected_ratio = avg_w / avg_h
    portrait_ratio = A4_SHORT / A4_LONG
    landscape_ratio = A4_LONG / A4_SHORT

    portrait_error = abs(detected_ratio - portrait_ratio) / portrait_ratio
    landscape_error = abs(detected_ratio - landscape_ratio) / landscape_ratio

    if portrait_error <= landscape_error:
        best_error = portrait_error
        out_w, out_h = A4_SHORT, A4_LONG
    else:
        best_error = landscape_error
        out_w, out_h = A4_LONG, A4_SHORT

    if best_error > 0.35:
        return (True, None, None)

    return (False, out_w, out_h)


def draw_preview_polyline(image, ordered_pts):
    preview = image.copy()
    poly = ordered_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(preview, [poly], isClosed=True, color=(0, 255, 0), thickness=6)
    return preview


def warp_to_a4(image, ordered_pts, out_w, out_h):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    return warped


def scanner_filter(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        251,
        11,
    )


def _build_inference_variants(image):
    variants = [image]

    # CLAHE-enhanced luminance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    variants.append(cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR))

    # mild sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    variants.append(cv2.filter2D(image, -1, kernel))

    # gamma brighten/darken variants for uneven lighting
    for gamma in (0.85, 1.20):
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]).astype("uint8")
        variants.append(cv2.LUT(image, table))

    return variants


def _score_selection(best_by_class):
    if not best_by_class:
        return 0.0
    return float(len(best_by_class)) * 1000.0 + sum(float(v.get("confidence", 0.0)) for v in best_by_class.values())


def run_inference_best(image_path):
    api_key = (os.getenv("ROBOFLOW_API_KEY") or "").strip()
    model_id = (os.getenv("ROBOFLOW_MODEL_ID") or "").strip()

    if not api_key or not model_id:
        return None, {"error": "MISSING_CONFIG", "required": ["ROBOFLOW_API_KEY", "ROBOFLOW_MODEL_ID"]}

    if InferenceHTTPClient is None:
        return None, {"error": "MISSING_DEPENDENCY", "required": "inference-sdk"}

    image = cv2.imread(image_path)
    if image is None:
        return None, {"error": "IMAGE_READ_ERROR"}

    h, w = image.shape[:2]
    primary_threshold = _env_float("OMR_CORNER_CONFIDENCE", DEFAULT_CONFIDENCE_THRESHOLD)
    fallback_threshold = _env_float("OMR_CORNER_CONFIDENCE_FALLBACK", DEFAULT_LOW_CONFIDENCE_THRESHOLD)

    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)

    best_global = {}
    best_score = -1.0
    used_threshold = primary_threshold
    used_variant = -1

    for variant_idx, variant in enumerate(_build_inference_variants(image)):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
            if not cv2.imwrite(temp_path, variant):
                continue

            result = client.infer(temp_path, model_id=model_id)
            predictions = result.get("predictions", [])

            selected = select_best_predictions(predictions, primary_threshold)
            threshold_used = primary_threshold

            if len(selected) < 4 and fallback_threshold < primary_threshold:
                selected_fb = select_best_predictions(predictions, fallback_threshold)
                if _score_selection(selected_fb) > _score_selection(selected):
                    selected = selected_fb
                    threshold_used = fallback_threshold

            score = _score_selection(selected)
            if score > best_score:
                best_score = score
                best_global = selected
                used_threshold = threshold_used
                used_variant = variant_idx

            if len(best_global) >= 4:
                break
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    if len(best_global) < 4:
        recovered, recovered_ok = _recover_missing_corner(best_global, w, h)
        if recovered_ok:
            best_global = recovered

    if len(best_global) < 4:
        return None, {
            "error": "INCOMPLETE_DETECTION",
            "details": {
                "detected_classes": sorted(list(best_global.keys())),
                "threshold_used": used_threshold,
                "variant_used": used_variant,
            },
        }

    raw_pts = [center_from_prediction(best_global[name]) for name in REQUIRED_CLASSES]
    ordered = order_points_clockwise(raw_pts)

    min_area_ratio = _env_float("OMR_MIN_POLYGON_AREA_RATIO", DEFAULT_MIN_POLYGON_AREA_RATIO)
    strong_corner_conf = _env_float("OMR_STRONG_CORNER_CONFIDENCE", DEFAULT_STRONG_CORNER_CONFIDENCE)
    area = _polygon_area(ordered)
    min_conf = _min_corner_confidence(best_global)

    has_inferred_corner = any(bool(v.get("inferred")) for v in best_global.values())

    # Only hard-fail small polygons when detections are weak or inferred.
    # If all 4 real corners are detected, allow processing to continue.
    if area < float(w * h) * min_area_ratio and (has_inferred_corner or min_conf < strong_corner_conf):
        return None, {
            "error": "INCOMPLETE_DETECTION",
            "details": {
                "reason": "SMALL_POLYGON",
                "polygon_area": area,
                "min_area": float(w * h) * min_area_ratio,
                "min_corner_confidence": min_conf,
                "strong_corner_confidence": strong_corner_conf,
                "has_inferred_corner": has_inferred_corner,
                "variant_used": used_variant,
            },
        }

    return {
        "corners": best_global,
        "threshold_used": used_threshold,
        "variant_used": used_variant,
        "recovered_missing": any(bool(v.get("inferred")) for v in best_global.values()),
    }, None


def process_document_result(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "IMAGE_READ_ERROR"}

    inference_pack, infer_error = run_inference_best(image_path)
    if infer_error is not None:
        return infer_error

    best_by_class = inference_pack["corners"]

    raw_pts = [center_from_prediction(best_by_class[name]) for name in REQUIRED_CLASSES]
    ordered_pts = order_points_clockwise(raw_pts)

    is_invalid, out_w, out_h = validate_angle_and_get_size(ordered_pts)
    if is_invalid:
        return {"error": "INVALID_ANGLE"}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = resolve_output_dir(script_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    preview_path = os.path.join(output_dir, f"{base_name}_preview.jpg")
    warp_debug_path = os.path.join(output_dir, f"{base_name}_warp_debug.jpg")
    processed_path = os.path.join(output_dir, f"{base_name}_processed.png")

    preview = draw_preview_polyline(image, ordered_pts)
    cv2.imwrite(preview_path, preview)

    warped = warp_to_a4(image, ordered_pts, out_w, out_h)
    cv2.imwrite(warp_debug_path, warped)

    processed = scanner_filter(warped)
    cv2.imwrite(processed_path, processed)

    bubble_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    bubble_result = analyze_bubbles(bubble_input, os.path.basename(processed_path))
    if "error" in bubble_result:
        return {"error": "BUBBLE_PROCESSING_FAILED", "details": bubble_result}

    corner_debug = {
        "required_order": REQUIRED_CLASSES,
        "raw_by_class": {
            cls: {
                "x": float(best_by_class[cls]["x"]),
                "y": float(best_by_class[cls]["y"]),
                "confidence": float(best_by_class[cls].get("confidence", 0.0)),
                "inferred": bool(best_by_class[cls].get("inferred", False)),
            }
            for cls in REQUIRED_CLASSES
        },
        "ordered_points": {
            "top_left": [float(ordered_pts[0][0]), float(ordered_pts[0][1])],
            "top_right": [float(ordered_pts[1][0]), float(ordered_pts[1][1])],
            "bottom_right": [float(ordered_pts[2][0]), float(ordered_pts[2][1])],
            "bottom_left": [float(ordered_pts[3][0]), float(ordered_pts[3][1])],
        },
        "variant_used": int(inference_pack.get("variant_used", -1)),
        "threshold_used": float(inference_pack.get("threshold_used", DEFAULT_CONFIDENCE_THRESHOLD)),
        "recovered_missing": bool(inference_pack.get("recovered_missing", False)),
    }

    return {
        "file": os.path.basename(image_path),
        "processed_path": processed_path,
        "preview_path": preview_path,
        "warp_debug_path": warp_debug_path,
        "bubble_preprocessed_path": bubble_result.get("preprocessed"),
        "debug": bubble_result.get("debug"),
        "sheet_id": bubble_result.get("sheet_id"),
        "answers": bubble_result.get("answers", {}),
        "corner_debug": corner_debug,
    }


def detect_bubble_grid(img, filename):
    temp_path = None
    try:
        ext = os.path.splitext(filename or "capture.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            temp_path = tmp.name

        if not cv2.imwrite(temp_path, img):
            return {"error": "IMAGE_WRITE_ERROR"}

        return process_document_result(temp_path)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def main():
    try:
        if len(sys.argv) < 2:
            print(json.dumps({"error": "NO_IMAGE_PATH"}))
            return

        result = process_document_result(sys.argv[1])
        print(json.dumps(result))
    except Exception:
        print(json.dumps({"error": "PROCESSING_FAILED", "traceback": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    main()
