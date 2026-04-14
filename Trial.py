import cv2
import numpy as np
import sys
import os
import json
import traceback
from pyzbar.pyzbar import decode


# ---------------- QR FALLBACK ----------------
def decode_qr_opencv(img):
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if data:
        return data.strip()
    return None


# ---------------- MAIN DETECTOR ----------------
def detect_bubble_grid(img, filename):
    try:
        # rotate sheet
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # preprocessing
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # ---------------- QR DETECTION ----------------
        qr_data = None
        h, w = img.shape[:2]

        qr_crop = img[
            int(0.02 * h):int(0.35 * h),
            int(0.02 * w):int(0.35 * w)
        ]

        codes = decode(qr_crop)
        if codes:
            qr_data = codes[0].data.decode("utf-8").strip()
        else:
            qr_data = decode_qr_opencv(qr_crop)

        # ---------------- FIND GRID ----------------
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        grid_bbox = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                xg, yg, wg, hg = cv2.boundingRect(approx)
                area = wg * hg
                if area > 0.02 * gray.shape[0] * gray.shape[1]:
                    grid_bbox = (xg, yg, wg, hg)
                    break

        # fallback grid
        if grid_bbox is None:
            h_full, w_full = gray.shape
            xg = int(w_full * 0.08)
            yg = int(h_full * 0.25)
            wg = int(w_full * 0.82)
            hg = int(h_full * 0.55)
            grid_bbox = (xg, yg, wg, hg)

        xg, yg, wg, hg = grid_bbox
        debug_img = img.copy()

        # ---------------- CONFIG ----------------
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

            col_gray = gray[yg:yg + hg, col_x1:col_x2]

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
                        candidates.append(
                            (cx, cy, bw, bh, bx, by, bx + bw, by + bh))
                return candidates

            candidates = detect_bubbles(col_gray, approx_bubble_w)

            candidates_full = [
                (
                    cx + col_x1, cy + yg, bw, bh,
                    x1 + col_x1, y1 + yg, x2 + col_x1, y2 + yg
                )
                for (cx, cy, bw, bh, x1, y1, x2, y2) in candidates
            ]

            # fallback grid boxes if detection fails
            if len(candidates_full) < rows * choices * 0.6:
                opt_w = col_w / choices
                for row in range(rows):
                    y1 = yg + int(row * row_height)
                    y2 = yg + int((row + 1) * row_height)
                    for opt in range(choices):
                        ox1 = int(col_x1 + opt * opt_w)
                        ox2 = int(col_x1 + (opt + 1) * opt_w)
                        green_boxes.append(
                            (ox1, y1, ox2, y2, row, opt, col))
                continue

            # group rows
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

            # fix row count
            while len(rows_groups) > rows:
                gaps = [(i, abs(np.mean([c[1] for c in rows_groups[i + 1]]) -
                                np.mean([c[1] for c in rows_groups[i]])))
                        for i in range(len(rows_groups) - 1)]

                merge_idx = min(gaps, key=lambda x: x[1])[0]
                rows_groups[merge_idx] += rows_groups.pop(merge_idx + 1)

            while len(rows_groups) < rows:
                spans = [max([c[1] for c in g]) - min([c[1] for c in g])
                         for g in rows_groups]

                idx = int(np.argmax(spans))
                group = sorted(rows_groups.pop(idx), key=lambda c: c[0])
                mid = len(group) // 2
                rows_groups.insert(idx, group[:mid])
                rows_groups.insert(idx + 1, group[mid:])

            # store boxes
            for r_idx, group in enumerate(rows_groups[:rows]):
                group_sorted = sorted(group, key=lambda c: c[0])
                for opt_idx, sel in enumerate(group_sorted[:choices]):
                    sx1, sy1, sx2, sy2 = sel[4], sel[5], sel[6], sel[7]
                    green_boxes.append(
                        (sx1, sy1, sx2, sy2, r_idx, opt_idx, col))

                    cv2.rectangle(
                        debug_img, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

        # ---------------- ANSWER SCORING ----------------
        questions_dict = {}

        for (sx1, sy1, sx2, sy2, r_idx, opt_idx, col) in green_boxes:
            q_num = r_idx + 1 + col * rows

            roi = gray[sy1:sy2, sx1:sx2]

            if roi.size == 0:
                mean_intensity = 255
            else:
                mean_intensity = np.mean(roi)

            questions_dict.setdefault(q_num, []).append({
                "opt_idx": opt_idx,
                "coords": (sx1, sy1, sx2, sy2),
                "mean": mean_intensity
            })

        final_answers = {}

        for q_num, opts in questions_dict.items():

            sorted_opts = sorted(opts, key=lambda x: x["mean"])

            darkest = sorted_opts[0]
            second = sorted_opts[1]

            diff = second["mean"] - darkest["mean"]

            # ---- DECISION LOGIC ----
            if diff < 15:
                ans = "invalid"      # multiple shaded
            elif darkest["mean"] > 180:
                ans = "blank"        # nothing shaded
            else:
                ans = chr(65 + darkest["opt_idx"])

                sx1, sy1, sx2, sy2 = darkest["coords"]
                cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                cv2.circle(debug_img, (cx, cy), 7, (0, 0, 255), 2)

            final_answers[str(q_num)] = ans

        # ---------------- DEBUG SAVE ----------------
        script_dir = os.path.dirname(os.path.abspath(__file__))

        debug_folder = os.path.abspath(
            os.path.join(script_dir, "..", "public", "storage", "debug")
        )

        os.makedirs(debug_folder, exist_ok=True)

        debug_filename = "debug_" + filename
        debug_path = os.path.join(debug_folder, debug_filename)

        cv2.imwrite(debug_path, debug_img)

        relative = os.path.join("debug", debug_filename).replace("\\", "/")

        return {
            "file": filename,
            "sheet_id": qr_data,
            "answers": final_answers,
            "debug": relative
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


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
