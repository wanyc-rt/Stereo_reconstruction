from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import extract_two_planes


class PromptDetector:
    def detect(self, color: np.ndarray, depth_m: np.ndarray, xyz_map: np.ndarray, valid_mask: np.ndarray) -> Dict[str, object]:
        raise NotImplementedError


class HeuristicPlateDetector(PromptDetector):
    def __init__(self, plane_distance_threshold_m: float, min_plane_points: int) -> None:
        self.plane_distance_threshold_m = plane_distance_threshold_m
        self.min_plane_points = min_plane_points

    def detect(self, color: np.ndarray, depth_m: np.ndarray, xyz_map: np.ndarray, valid_mask: np.ndarray) -> Dict[str, object]:
        planes = extract_two_planes(
            xyz_map,
            valid_mask,
            threshold=self.plane_distance_threshold_m,
            min_points=self.min_plane_points,
        )
        if len(planes) < 2:
            return {"plates": [], "roi_mask": np.zeros(depth_m.shape, dtype=bool)}

        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        edge_mask = cv2.Canny(gray, 60, 150) > 0
        low_sat = hsv[:, :, 1] < np.percentile(hsv[:, :, 1], 60)
        mid_value = hsv[:, :, 2] > np.percentile(hsv[:, :, 2], 20)
        color_plate_prior = low_sat & mid_value

        refined = []
        for plane in planes[:2]:
            mask = plane["mask"].copy()
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2).astype(bool)
            expanded = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
            mask = mask | (edge_mask & expanded) | (color_plate_prior & expanded)
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0, 0, 0, 0]
            plane["mask"] = mask
            plane["bbox"] = bbox
            refined.append(plane)

        roi = refined[0]["mask"] | refined[1]["mask"]
        return {"plates": refined, "roi_mask": roi, "color_plate_prior": color_plate_prior}


class OpenVocabularyDetector(PromptDetector):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "google/owlvit-base-patch32"
        self._detector = None
        try:
            from transformers import pipeline
            self._detector = pipeline("zero-shot-object-detection", model=self.model_name)
        except Exception:
            self._detector = None

    def detect(self, color: np.ndarray, depth_m: np.ndarray, xyz_map: np.ndarray, valid_mask: np.ndarray) -> Dict[str, object]:
        if self._detector is None:
            return {"plates": [], "roi_mask": np.zeros(depth_m.shape, dtype=bool), "backend": "unavailable"}
        labels = ["steel plate", "metal plate", "weld seam", "groove", "joint"]
        results = self._detector(color, candidate_labels=labels)
        roi_mask = np.zeros(depth_m.shape, dtype=bool)
        boxes: List[Dict[str, object]] = []
        for item in results:
            box = item.get("box", {})
            x0 = int(box.get("xmin", 0))
            y0 = int(box.get("ymin", 0))
            x1 = int(box.get("xmax", 0))
            y1 = int(box.get("ymax", 0))
            if x1 <= x0 or y1 <= y0:
                continue
            roi_mask[y0:y1, x0:x1] = True
            boxes.append({"label": item.get("label", "object"), "score": float(item.get("score", 0.0)), "bbox": [x0, y0, x1, y1]})
        return {"plates": boxes, "roi_mask": roi_mask, "backend": "open_vocabulary"}
