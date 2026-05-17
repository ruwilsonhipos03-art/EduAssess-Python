import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np


ITEMS_PER_COL = 25
CHOICES = ["A", "B", "C", "D", "E"]
COLS = 4


def _cluster_axis(values: List[float], max_gap: float) -> List[List[float]]:
    clusters: List[List[float]] = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > max_gap:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def _center(cluster: List[float]) -> float:
    return float(np.mean(cluster))


def _ratio_in_circle(thresh_roi: np.ndarray, cx: float, cy: float, radius: int) -> float:
    mask = np.zeros(thresh_roi.shape, dtype="uint8")
    cv2.circle(mask, (int(round(cx)), int(round(cy))), radius, 255, -1)
    total_pixels = np.sum(mask == 255)
    if total_pixels == 0:
        return 0.0
    return float(np.sum((thresh_roi == 255) & (mask == 255)) / total_pixels)


def detect_bubble_grid(thresh_roi: np.ndarray, filename: str) -> Dict:
    """
    Analyzes the Image 03 binary matrix by learning the printed OMR grid.
    The final read uses expected row/choice centers, so a weak or broken
    bubble contour does not make the whole row invalid.
    """
    if len(thresh_roi.shape) == 3:
        thresh_roi = cv2.cvtColor(thresh_roi, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(thresh_roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return {"error": "No structural contours detected in bubble workspace."}

    valid_bubbles = []
    min_area, max_area = 50, 600

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue

        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)

        if min_area < area < max_area and 0.70 <= aspect_ratio <= 1.40:
            M = cv2.moments(c)
            if M["m00"] != 0:
                valid_bubbles.append({
                    "cx": int(M["m10"] / M["m00"]),
                    "cy": int(M["m01"] / M["m00"]),
                    "w": w,
                    "h": h,
                })

    if len(valid_bubbles) < 300:
        return {"error": f"Dynamic layout sorting failed. Found only {len(valid_bubbles)} candidate circles."}

    avg_radius = int(np.median([max(b["w"], b["h"]) / 2 for b in valid_bubbles]))
    avg_radius = max(5, avg_radius)

    x_clusters = _cluster_axis([b["cx"] for b in valid_bubbles], max(avg_radius * 1.6, 16))
    min_track_hits = max(10, int(ITEMS_PER_COL * 0.65))
    x_tracks = [_center(cluster) for cluster in x_clusters if len(cluster) >= min_track_hits]

    if len(x_tracks) < COLS * len(CHOICES):
        min_track_hits = max(6, int(ITEMS_PER_COL * 0.45))
        x_tracks = [_center(cluster) for cluster in x_clusters if len(cluster) >= min_track_hits]

    x_tracks = sorted(x_tracks)[: COLS * len(CHOICES)]
    if len(x_tracks) != COLS * len(CHOICES):
        return {"error": f"Expected 20 answer tracks but found {len(x_tracks)}."}

    final_answers = {}
    bubble_telemetry = []
    fill_radius = max(3, int(avg_radius * 0.62))

    for col_idx in range(COLS):
        col_x_tracks = x_tracks[col_idx * len(CHOICES) : (col_idx + 1) * len(CHOICES)]
        left_bound = min(col_x_tracks) - (avg_radius * 1.8)
        right_bound = max(col_x_tracks) + (avg_radius * 1.8)
        col_points = [b for b in valid_bubbles if left_bound <= b["cx"] <= right_bound]

        y_clusters = _cluster_axis([b["cy"] for b in col_points], max(avg_radius * 1.35, 14))
        y_tracks = [_center(cluster) for cluster in y_clusters if len(cluster) >= 3]
        y_tracks = sorted(y_tracks)[:ITEMS_PER_COL]

        if len(y_tracks) < ITEMS_PER_COL:
            return {"error": f"Expected 25 rows in column {col_idx + 1} but found {len(y_tracks)}."}

        for row_idx, row_y in enumerate(y_tracks):
            item_num = row_idx + 1 + (col_idx * ITEMS_PER_COL)
            key = str(item_num)

            ratios = [
                _ratio_in_circle(thresh_roi, choice_x, row_y, fill_radius)
                for choice_x in col_x_tracks
            ]
            ordered = sorted(ratios, reverse=True)
            strongest = ordered[0]
            runner_up = ordered[1] if len(ordered) > 1 else 0.0

            detected_filled_for_row = []
            row_telemetry_cache = []

            for choice_idx, (choice_x, fill_ratio) in enumerate(zip(col_x_tracks, ratios)):
                is_filled = False
                if strongest >= 0.68:
                    if fill_ratio == strongest and strongest >= runner_up + 0.10:
                        is_filled = True
                    elif fill_ratio >= 0.68 and fill_ratio >= strongest - 0.03:
                        is_filled = True

                row_telemetry_cache.append(((int(round(choice_x)), int(round(row_y))), is_filled, key))
                if is_filled:
                    detected_filled_for_row.append(CHOICES[choice_idx])

            bubble_telemetry.extend(row_telemetry_cache)

            if len(detected_filled_for_row) == 1:
                final_answers[key] = detected_filled_for_row[0]
            elif len(detected_filled_for_row) > 1:
                final_answers[key] = "multimark"
            else:
                final_answers[key] = "blank"

    return {
        "answers": final_answers,
        "bubble_centers": bubble_telemetry,
        "calculated_radius": avg_radius,
    }


def analyze_bubbles(thresh_roi: np.ndarray, filename: str) -> Dict:
    return detect_bubble_grid(thresh_roi, filename)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "NO_IMAGE_PATH"}))
        return
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(json.dumps({"error": "IMAGE_READ_ERROR"}))
        return
    print(json.dumps(detect_bubble_grid(img, os.path.basename(sys.argv[1]))))


if __name__ == "__main__":
    main()
