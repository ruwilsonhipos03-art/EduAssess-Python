import cv2
import numpy as np
import sys
import os
import json
import traceback
try:
    from pyzbar.pyzbar import decode as decode_pyzbar
except Exception:
    decode_pyzbar = None


def resolve_debug_folder(script_dir):
    configured = (os.getenv("OMR_DEBUG_DIR") or "").strip()
    if configured:
        return os.path.abspath(configured)

    return os.path.abspath(
        os.path.join(script_dir, "omr","output")
    )


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    max_width = max(max_width, 10)
    max_height = max(max_height, 10)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


# ---------------- QR FALLBACK ----------------
def decode_qr_opencv(img):
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        return data.strip()
    return None


def find_answer_grid_bbox(gray):
    """Locate the big answer-grid rectangle in the lower part of the page."""
    h, w = gray.shape[:2]

    # Focus only on lower sheet area to avoid QR/logo/header noise.
    y_start = int(0.28 * h)
    lower = gray[y_start:, :]

    blur = cv2.GaussianBlur(lower, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.10 * w * h:
            continue
        if ww < 0.60 * w:
            continue
        if hh < 0.35 * h:
            continue
        candidates.append((area, x, y + y_start, ww, hh))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, x, y, ww, hh = candidates[0]
        return (x, y, ww, hh)

    # Fallback tuned for this answer sheet layout.
    fx = int(0.03 * w)
    fy = int(0.30 * h)
    fw = int(0.94 * w)
    fh = int(0.62 * h)
    return (fx, fy, fw, fh)


# ---------------- MAIN DETECTOR ----------------
def detect_bubble_grid(img, filename):
    try:
        # keep rotation
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # ---------------- QR DETECTION ----------------
        qr_data = None
        h, w = img.shape[:2]

        qr_crop = img[
            int(0.02 * h):int(0.35 * h),
            int(0.02 * w):int(0.35 * w)
        ]

        codes = decode_pyzbar(qr_crop) if decode_pyzbar else []
        if codes:
            qr_data = codes[0].data.decode("utf-8").strip()
        else:
            qr_data = decode_qr_opencv(qr_crop)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect sheet boundary for perspective warp
        edges = cv2.Canny(gray, 75, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sheet_quad = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, ww, hh = cv2.boundingRect(approx)
                if ww * hh > 0.30 * w * h:
                    sheet_quad = approx.reshape(4, 2).astype("float32")
                    break

        if sheet_quad is not None:
            warp_img = four_point_warp(img, sheet_quad)
        else:
            warp_img = img.copy()

        before_check_img = warp_img.copy()
        warp_gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)

        # ---------------- GRID REGION ----------------
        # Restrict bubble scan to the answer-grid rectangle only.
        xg, yg, wg, hg = find_answer_grid_bbox(warp_gray)
        green_rect_img = warp_img.copy()
        debug_img = warp_img.copy()

        cols = 4
        rows = 25
        choices = 5

        col_width = wg / cols
        row_height = hg / rows
        green_boxes = []

        # ---------------- BUBBLE DETECTION ----------------
        for col in range(cols):
            col_x1 = xg + int(col * col_width)
            col_x2 = xg + int((col + 1) * col_width)
            col_w = col_x2 - col_x1

            col_gray = warp_gray[yg:yg + hg, col_x1:col_x2]
            approx_bubble_w = max(6, int((col_w / choices) * 0.9))

            def detect_bubbles(sub_gray, approx_d):
                loc_blur = cv2.GaussianBlur(sub_gray, (5, 5), 0)

                loc_thresh = cv2.adaptiveThreshold(
                    loc_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 15, 6
                )

                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                loc_closed = cv2.morphologyEx(
                    loc_thresh, cv2.MORPH_CLOSE, kern, iterations=2)

                cnts, _ = cv2.findContours(
                    loc_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                candidates = []
                for c in cnts:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    min_d = max(4, int(approx_d * 0.5))
                    max_d = max(6, int(approx_d * 2.2))

                    if min_d <= bw <= max_d and min_d <= bh <= max_d and bw * bh > 10:
                        cx = bx + bw // 2
                        cy = by + bh // 2
                        candidates.append((cx, cy, bw, bh, bx, by, bx + bw, by + bh))
                return candidates

            candidates = detect_bubbles(col_gray, approx_bubble_w)

            candidates_full = [
                (
                    cx + col_x1, cy + yg, bw, bh,
                    x1 + col_x1, y1 + yg, x2 + col_x1, y2 + yg
                )
                for (cx, cy, bw, bh, x1, y1, x2, y2) in candidates
            ]

            if len(candidates_full) < rows * choices * 0.6:
                opt_w = col_w / choices
                for row in range(rows):
                    y1 = yg + int(row * row_height)
                    y2 = yg + int((row + 1) * row_height)
                    for opt in range(choices):
                        ox1 = int(col_x1 + opt * opt_w)
                        ox2 = int(col_x1 + (opt + 1) * opt_w)
                        green_boxes.append((ox1, y1, ox2, y2, row, opt, col))
                continue

            candidates_full.sort(key=lambda c: c[1])
            rows_groups = []
            current = [candidates_full[0]]

            for cand in candidates_full[1:]:
                prev_y = np.mean([c[1] for c in current])
                if abs(cand[1] - prev_y) <= max(8, row_height * 0.4):
                    current.append(cand)
                else:
                    rows_groups.append(current)
                    current = [cand]

            rows_groups.append(current)

            while len(rows_groups) > rows:
                gaps = [
                    (i, abs(np.mean([c[1] for c in rows_groups[i + 1]]) - np.mean([c[1] for c in rows_groups[i]])))
                    for i in range(len(rows_groups) - 1)
                ]
                merge_idx = min(gaps, key=lambda x: x[1])[0]
                rows_groups[merge_idx] += rows_groups.pop(merge_idx + 1)

            while len(rows_groups) < rows:
                spans = [max([c[1] for c in g]) - min([c[1] for c in g]) for g in rows_groups]
                idx = int(np.argmax(spans))
                group = sorted(rows_groups.pop(idx), key=lambda c: c[0])
                mid = len(group) // 2
                rows_groups.insert(idx, group[:mid])
                rows_groups.insert(idx + 1, group[mid:])

            for r_idx, group in enumerate(rows_groups[:rows]):
                group_sorted = sorted(group, key=lambda c: c[0])
                for opt_idx, sel in enumerate(group_sorted[:choices]):
                    sx1, sy1, sx2, sy2 = sel[4], sel[5], sel[6], sel[7]
                    green_boxes.append((sx1, sy1, sx2, sy2, r_idx, opt_idx, col))

        questions_dict = {}

        for (sx1, sy1, sx2, sy2, r_idx, opt_idx, col) in green_boxes:
            cv2.rectangle(green_rect_img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

            q_num = r_idx + 1 + col * rows
            roi = warp_gray[sy1:sy2, sx1:sx2]
            mean_intensity = 255 if roi.size == 0 else np.mean(roi)

            questions_dict.setdefault(q_num, []).append({
                "opt_idx": opt_idx,
                "coords": (sx1, sy1, sx2, sy2),
                "mean": mean_intensity
            })

        final_answers = {}

        for q_num, opts in questions_dict.items():
            if len(opts) < 2:
                final_answers[str(q_num)] = "invalid"
                continue

            sorted_opts = sorted(opts, key=lambda x: x["mean"])
            darkest = sorted_opts[0]
            second = sorted_opts[1]
            diff = second["mean"] - darkest["mean"]

            if diff < 15:
                ans = "invalid"
            elif darkest["mean"] > 180:
                ans = "blank"
            else:
                ans = chr(65 + darkest["opt_idx"])
                sx1, sy1, sx2, sy2 = darkest["coords"]
                cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                cv2.circle(debug_img, (cx, cy), 7, (0, 0, 255), 2)

            final_answers[str(q_num)] = ans

        # ---------------- DEBUG SAVE (EXACTLY 4 IMAGES) ----------------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        debug_folder = resolve_debug_folder(script_dir)
        os.makedirs(debug_folder, exist_ok=True)

        name_no_ext, ext = os.path.splitext(filename)
        if not ext:
            ext = ".jpg"

        files = {
            "green_rectangles": f"{name_no_ext}_01_green_rectangles{ext}",
            "perspective_warp": f"{name_no_ext}_02_perspective_warp{ext}",
            "before_check_bubbles": f"{name_no_ext}_03_before_check_bubbles{ext}",
            "after_check_bubbles": f"{name_no_ext}_04_after_check_bubbles{ext}"
        }

        cv2.imwrite(os.path.join(debug_folder, files["green_rectangles"]), green_rect_img)
        cv2.imwrite(os.path.join(debug_folder, files["perspective_warp"]), warp_img)
        cv2.imwrite(os.path.join(debug_folder, files["before_check_bubbles"]), before_check_img)
        cv2.imwrite(os.path.join(debug_folder, files["after_check_bubbles"]), debug_img)

        relative_debug = {
            key: os.path.join("omr_processed", value).replace("\\", "/")
            for key, value in files.items()
        }

        return {
            "file": filename,
            "sheet_id": qr_data,
            "answers": final_answers,
            # Backward-compatible field expected by Laravel (string path).
            "debug": relative_debug["after_check_bubbles"],
            # Full set of debug artifacts.
            "debug_images": relative_debug
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# Backward-compatible entrypoint expected by CheckExam.py
def analyze_bubbles(img, filename):
    return detect_bubble_grid(img, filename)


# ---------------- MAIN ----------------
def main():
    try:
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
            img = cv2.imread(img_path)

            if img is None:
                print(json.dumps({"error": "Cannot read image"}))
                return

            filename = os.path.basename(img_path)
            result = detect_bubble_grid(img, filename)
            print(json.dumps(result))
        else:
            print(json.dumps({"error": "No image provided"}))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
