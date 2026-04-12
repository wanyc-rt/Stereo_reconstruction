import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np

from .geometry import fit_plane_ransac
from .io_utils import ensure_dir


_CANDIDATE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class SegmentCandidate:
    candidate_id: str
    candidate_source: str
    mask: np.ndarray
    bbox: List[int]
    area_px: int
    valid_depth_ratio: float
    plane_inlier_ratio: float
    border_touch_ratio: float
    bbox_fill_ratio: float
    image_area_ratio: float
    centroid_uv: List[float]
    center_distance_norm: float
    plane_normal: Optional[List[float]]
    plane_offset_m: Optional[float]
    normal_z_abs: float
    geometry_score: float
    sam_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_source": self.candidate_source,
            "bbox": self.bbox,
            "area_px": self.area_px,
            "valid_depth_ratio": self.valid_depth_ratio,
            "plane_inlier_ratio": self.plane_inlier_ratio,
            "border_touch_ratio": self.border_touch_ratio,
            "bbox_fill_ratio": self.bbox_fill_ratio,
            "image_area_ratio": self.image_area_ratio,
            "centroid_uv": self.centroid_uv,
            "center_distance_norm": self.center_distance_norm,
            "plane_normal": self.plane_normal,
            "plane_offset_m": self.plane_offset_m,
            "normal_z_abs": self.normal_z_abs,
            "geometry_score": self.geometry_score,
            "sam_score": self.sam_score,
        }


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float((mask_a & mask_b).sum())
    union = float((mask_a | mask_b).sum())
    return inter / max(union, 1.0)


def _mask_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask & (~eroded)


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_border_touch_ratio(mask: np.ndarray) -> float:
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(border.mean()) if len(border) else 0.0


def _mask_fill_ratio(mask: np.ndarray, bbox: Sequence[int]) -> float:
    x0, y0, x1, y1 = bbox
    box_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
    return float(mask.sum() / box_area)


def _band_linearity(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 32:
        return 0.0, 0.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    major = float(s[0]) if len(s) > 0 else 0.0
    minor = float(s[1]) if len(s) > 1 else 0.0
    elongation = major / max(minor, 1e-6)
    linearity = float(np.clip((elongation - 2.0) / 10.0, 0.0, 1.0))
    return linearity, float(elongation)


def _geometry_score(
    area_ratio: float,
    border_touch_ratio: float,
    valid_depth_ratio: float,
    plane_inlier_ratio: float,
    fill_ratio: float,
    center_distance_norm: float,
    normal_z_abs: float,
) -> float:
    area_term = min(area_ratio / 0.18, 1.0)
    area_penalty = max(area_ratio - 0.55, 0.0) * 2.0
    return (
        1.40 * area_term
        + 1.20 * valid_depth_ratio
        + 1.10 * plane_inlier_ratio
        + 0.65 * fill_ratio
        + 0.35 * normal_z_abs
        - 1.95 * border_touch_ratio
        - 0.85 * center_distance_norm
        - area_penalty
    )


class SAMSegmenter:
    def __init__(self, python_executable: str, model_cfg: str, checkpoint_path: str, max_masks: int) -> None:
        self.python_executable = python_executable
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.max_masks = max_masks
        self.helper_script = os.path.join(os.path.dirname(__file__), "sam_infer.py")

    def is_available(self) -> bool:
        return os.path.exists(self.python_executable) and os.path.exists(self.helper_script) and os.path.exists(self.checkpoint_path)

    def generate(self, color: np.ndarray, work_dir: str, frame_id: str) -> List[Dict[str, object]]:
        if not self.is_available():
            return []
        ensure_dir(work_dir)
        image_path = os.path.join(work_dir, f"{frame_id}_sam_input.png")
        npz_path = os.path.join(work_dir, f"{frame_id}_sam_masks.npz")
        json_path = os.path.join(work_dir, f"{frame_id}_sam_meta.json")
        imageio.imwrite(image_path, color)
        cmd = [
            self.python_executable,
            self.helper_script,
            "--image_path",
            image_path,
            "--output_npz",
            npz_path,
            "--output_json",
            json_path,
            "--model_cfg",
            self.model_cfg,
            "--checkpoint_path",
            self.checkpoint_path,
            "--max_masks",
            str(self.max_masks),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(npz_path) or not os.path.exists(json_path):
            return []
        payload = json.load(open(json_path, "r", encoding="utf-8"))
        masks = np.load(npz_path)["masks"]
        out: List[Dict[str, object]] = []
        for meta, mask in zip(payload.get("masks", []), masks):
            out.append(
                {
                    "mask": mask.astype(bool),
                    "bbox": meta.get("bbox", [0, 0, 0, 0]),
                    "area": int(meta.get("area", int(mask.sum()))),
                    "predicted_iou": float(meta.get("predicted_iou", 0.0)),
                    "stability_score": float(meta.get("stability_score", 0.0)),
                }
            )
        return out


def build_segment_candidates(
    segment_masks: List[Dict[str, object]],
    xyz_map: np.ndarray,
    valid_mask: np.ndarray,
    min_mask_area_px: int,
    plane_distance_threshold_m: float,
    min_plane_points: int,
) -> List[SegmentCandidate]:
    h, w = valid_mask.shape
    candidates: List[SegmentCandidate] = []
    for idx, item in enumerate(segment_masks):
        if idx >= len(_CANDIDATE_LABELS):
            break
        mask = item["mask"].astype(bool)
        area_px = int(mask.sum())
        if area_px < min_mask_area_px:
            continue
        bbox = _bbox_from_mask(mask)
        valid_points = mask & valid_mask
        valid_depth_ratio = float(valid_points.sum() / max(area_px, 1))
        if valid_points.sum() < max(min_plane_points // 2, 1200):
            continue
        flat_points = xyz_map[valid_points]
        plane = fit_plane_ransac(flat_points, threshold=plane_distance_threshold_m, min_points=max(min_plane_points // 2, 1200))
        plane_inlier_ratio = 0.0
        plane_normal: Optional[List[float]] = None
        plane_offset_m: Optional[float] = None
        normal_z_abs = 0.0
        if plane is not None:
            plane_inlier_ratio = float(len(plane["inliers"]) / max(len(flat_points), 1))
            plane_normal = plane["normal"].astype(np.float32).tolist()
            plane_offset_m = float(plane["model"][3])
            normal_z_abs = abs(float(plane["normal"][2]))
        border_touch_ratio = _mask_border_touch_ratio(mask)
        fill_ratio = _mask_fill_ratio(mask, bbox)
        area_ratio = float(area_px / float(h * w))
        ys, xs = np.where(mask)
        centroid_uv = [float(xs.mean()), float(ys.mean())]
        dx = (centroid_uv[0] - 0.5 * w) / max(0.5 * w, 1.0)
        dy = (centroid_uv[1] - 0.5 * h) / max(0.5 * h, 1.0)
        center_distance_norm = float(np.sqrt(dx * dx + dy * dy))
        score = _geometry_score(
            area_ratio,
            border_touch_ratio,
            valid_depth_ratio,
            plane_inlier_ratio,
            fill_ratio,
            center_distance_norm,
            normal_z_abs,
        )
        sam_score = 0.5 * float(item.get("predicted_iou", 0.0)) + 0.5 * float(item.get("stability_score", 0.0))
        candidates.append(
            SegmentCandidate(
                candidate_id=_CANDIDATE_LABELS[idx],
                candidate_source="sam",
                mask=mask,
                bbox=bbox,
                area_px=area_px,
                valid_depth_ratio=valid_depth_ratio,
                plane_inlier_ratio=plane_inlier_ratio,
                border_touch_ratio=border_touch_ratio,
                bbox_fill_ratio=fill_ratio,
                image_area_ratio=area_ratio,
                centroid_uv=centroid_uv,
                center_distance_norm=center_distance_norm,
                plane_normal=plane_normal,
                plane_offset_m=plane_offset_m,
                normal_z_abs=normal_z_abs,
                geometry_score=score,
                sam_score=sam_score,
            )
        )
    candidates.sort(key=lambda item: item.geometry_score, reverse=True)
    return candidates


def build_geometry_candidates(
    plates: List[Dict[str, object]],
    valid_mask: np.ndarray,
) -> List[SegmentCandidate]:
    h, w = valid_mask.shape
    candidates: List[SegmentCandidate] = []
    for idx, plate in enumerate(plates):
        if idx >= len(_CANDIDATE_LABELS):
            break
        mask = np.asarray(plate.get("mask"), dtype=bool)
        if mask.shape != valid_mask.shape:
            continue
        area_px = int(mask.sum())
        if area_px <= 0:
            continue
        bbox = plate.get("bbox") or _bbox_from_mask(mask)
        valid_depth_ratio = float((mask & valid_mask).sum() / max(area_px, 1))
        border_touch_ratio = _mask_border_touch_ratio(mask)
        fill_ratio = _mask_fill_ratio(mask, bbox)
        area_ratio = float(area_px / float(h * w))
        ys, xs = np.where(mask)
        centroid_uv = [float(xs.mean()), float(ys.mean())] if len(xs) else [0.0, 0.0]
        dx = (centroid_uv[0] - 0.5 * w) / max(0.5 * w, 1.0)
        dy = (centroid_uv[1] - 0.5 * h) / max(0.5 * h, 1.0)
        center_distance_norm = float(np.sqrt(dx * dx + dy * dy))
        plane_normal = plate.get("plane_normal")
        plane_offset_m = plate.get("plane_offset_m")
        normal_z_abs = abs(float(plane_normal[2])) if isinstance(plane_normal, (list, tuple)) and len(plane_normal) >= 3 else 0.0
        plane_inlier_ratio = float(plate.get("plane_inlier_ratio", 1.0 if plane_normal is not None else 0.0))
        score = _geometry_score(
            area_ratio,
            border_touch_ratio,
            valid_depth_ratio,
            plane_inlier_ratio,
            fill_ratio,
            center_distance_norm,
            normal_z_abs,
        )
        candidates.append(
            SegmentCandidate(
                candidate_id=_CANDIDATE_LABELS[idx],
                candidate_source="geometry",
                mask=mask,
                bbox=[int(v) for v in bbox],
                area_px=area_px,
                valid_depth_ratio=valid_depth_ratio,
                plane_inlier_ratio=plane_inlier_ratio,
                border_touch_ratio=border_touch_ratio,
                bbox_fill_ratio=fill_ratio,
                image_area_ratio=area_ratio,
                centroid_uv=centroid_uv,
                center_distance_norm=center_distance_norm,
                plane_normal=list(plane_normal) if isinstance(plane_normal, (list, tuple)) else None,
                plane_offset_m=float(plane_offset_m) if plane_offset_m is not None else None,
                normal_z_abs=normal_z_abs,
                geometry_score=score,
                sam_score=0.0,
            )
        )
    candidates.sort(key=lambda item: item.geometry_score, reverse=True)
    return candidates


def relabel_candidates(candidates: List[SegmentCandidate]) -> List[SegmentCandidate]:
    relabeled: List[SegmentCandidate] = []
    for idx, candidate in enumerate(candidates):
        if idx >= len(_CANDIDATE_LABELS):
            break
        relabeled.append(
            SegmentCandidate(
                candidate_id=_CANDIDATE_LABELS[idx],
                candidate_source=candidate.candidate_source,
                mask=candidate.mask,
                bbox=list(candidate.bbox),
                area_px=candidate.area_px,
                valid_depth_ratio=candidate.valid_depth_ratio,
                plane_inlier_ratio=candidate.plane_inlier_ratio,
                border_touch_ratio=candidate.border_touch_ratio,
                bbox_fill_ratio=candidate.bbox_fill_ratio,
                image_area_ratio=candidate.image_area_ratio,
                centroid_uv=list(candidate.centroid_uv),
                center_distance_norm=candidate.center_distance_norm,
                plane_normal=list(candidate.plane_normal) if candidate.plane_normal is not None else None,
                plane_offset_m=candidate.plane_offset_m,
                normal_z_abs=candidate.normal_z_abs,
                geometry_score=candidate.geometry_score,
                sam_score=candidate.sam_score,
            )
        )
    return relabeled


def merge_candidate_sources(
    sam_candidates: List[SegmentCandidate],
    geometry_candidates: List[SegmentCandidate],
    max_candidates: int,
    dedup_iou_threshold: float = 0.72,
) -> List[SegmentCandidate]:
    merged: List[SegmentCandidate] = []
    ordered = list(sam_candidates) + list(geometry_candidates)
    ordered.sort(
        key=lambda item: (
            item.geometry_score,
            0.35 if item.candidate_source == "geometry" else 0.0,
            item.sam_score,
        ),
        reverse=True,
    )
    for candidate in ordered:
        duplicate = False
        for kept in merged:
            iou = _mask_iou(candidate.mask, kept.mask)
            if iou >= dedup_iou_threshold:
                duplicate = True
                if (
                    candidate.candidate_source == "geometry"
                    and kept.candidate_source == "sam"
                    and candidate.geometry_score >= kept.geometry_score - 0.05
                ):
                    kept.candidate_source = "sam+geometry"
                    kept.plane_normal = candidate.plane_normal or kept.plane_normal
                    kept.plane_offset_m = candidate.plane_offset_m if candidate.plane_offset_m is not None else kept.plane_offset_m
                    kept.plane_inlier_ratio = max(kept.plane_inlier_ratio, candidate.plane_inlier_ratio)
                    kept.geometry_score = max(kept.geometry_score, candidate.geometry_score)
                elif (
                    candidate.candidate_source == "sam"
                    and kept.candidate_source == "geometry"
                    and candidate.sam_score > 0.0
                ):
                    kept.candidate_source = "sam+geometry"
                    kept.sam_score = max(kept.sam_score, candidate.sam_score)
                break
        if duplicate:
            continue
        merged.append(candidate)
        if len(merged) >= max_candidates:
            break
    return relabel_candidates(merged)


def select_geometry_candidates(candidates: List[SegmentCandidate], top_k: int) -> List[SegmentCandidate]:
    if not candidates:
        return []
    chosen: List[SegmentCandidate] = []
    for candidate in candidates:
        if candidate.geometry_score < 0.8 and chosen:
            continue
        if candidate.border_touch_ratio > 0.18:
            continue
        overlaps = []
        for picked in chosen:
            inter = float((candidate.mask & picked.mask).sum())
            union = float((candidate.mask | picked.mask).sum())
            overlaps.append(inter / max(union, 1.0))
        if overlaps and max(overlaps) > 0.88:
            continue
        chosen.append(candidate)
        if len(chosen) >= top_k:
            break
    fallback = [candidate for candidate in candidates if candidate.border_touch_ratio <= 0.22]
    return chosen or (fallback[:1] if fallback else candidates[:1])


def _draw_candidate_outline(canvas: np.ndarray, candidate: SegmentCandidate, color: Sequence[int]) -> None:
    contours, _ = cv2.findContours(candidate.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
    x0, y0, _, _ = candidate.bbox
    cv2.putText(canvas, candidate.candidate_id, (x0 + 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def render_candidate_overlay(color: np.ndarray, candidates: List[SegmentCandidate]) -> np.ndarray:
    palette = [
        (255, 80, 80),
        (80, 220, 255),
        (80, 255, 120),
        (255, 180, 0),
        (255, 0, 180),
        (180, 255, 0),
    ]
    canvas = color.copy()
    for idx, candidate in enumerate(candidates):
        tint = np.asarray(palette[idx % len(palette)], dtype=np.float32)
        mask = candidate.mask
        canvas[mask] = (0.72 * canvas[mask] + 0.28 * tint).astype(np.uint8)
        _draw_candidate_outline(canvas, candidate, palette[idx % len(palette)])
    return canvas


def summarize_candidate_pairs(candidates: List[SegmentCandidate]) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            inter = float((a.mask & b.mask).sum())
            union = float((a.mask | b.mask).sum())
            iou = inter / max(union, 1.0)
            if iou > 0.80:
                continue
            edge_a = _mask_edge(a.mask)
            edge_b = _mask_edge(b.mask)
            dist_to_a = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, 5)
            dist_to_b = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, 5)
            near_band = (dist_to_a < 24.0) & (dist_to_b < 24.0) & (a.mask | b.mask)
            near_area = int(near_band.sum())
            band_linearity, band_elongation = _band_linearity(near_band)
            min_edge_gap_px = float(min(dist_to_b[edge_a].min(initial=np.inf), dist_to_a[edge_b].min(initial=np.inf)))
            normal_alignment = None
            if a.plane_normal is not None and b.plane_normal is not None:
                na = np.asarray(a.plane_normal, dtype=np.float32)
                nb = np.asarray(b.plane_normal, dtype=np.float32)
                normal_alignment = float(abs(np.dot(na, nb) / max(np.linalg.norm(na) * np.linalg.norm(nb), 1e-6)))
            pair_source_support = int("geometry" in a.candidate_source) + int("geometry" in b.candidate_source)
            seam_shape_hint = "curve" if band_linearity < 0.22 and band_elongation > 2.0 else "line"
            pair_score = (
                0.40 * np.clip(near_area / 18000.0, 0.0, 1.0)
                + 0.25 * (normal_alignment if normal_alignment is not None else 0.0)
                + 0.20 * band_linearity
                + 0.15 * np.clip((24.0 - min(min_edge_gap_px, 24.0)) / 24.0, 0.0, 1.0)
            )
            pairs.append(
                {
                    "pair": [a.candidate_id, b.candidate_id],
                    "iou": iou,
                    "near_band_area_px": near_area,
                    "min_edge_gap_px": None if not np.isfinite(min_edge_gap_px) else min_edge_gap_px,
                    "normal_alignment": normal_alignment,
                    "band_linearity": band_linearity,
                    "band_elongation": band_elongation,
                    "pair_source_support": pair_source_support,
                    "seam_shape_hint": seam_shape_hint,
                    "pair_score": float(pair_score),
                }
            )
    pairs.sort(key=lambda item: item["pair_score"], reverse=True)
    return pairs[:8]


def build_joint_roi_mask(selected: List[SegmentCandidate], max_gap_px: float = 24.0) -> Tuple[np.ndarray, Dict[str, object]]:
    if not selected:
        raise ValueError("selected must not be empty")
    if len(selected) == 1:
        return selected[0].mask.copy(), {"mode": "single_candidate"}
    a, b = selected[0], selected[1]
    edge_a = _mask_edge(a.mask)
    edge_b = _mask_edge(b.mask)
    dist_to_a = cv2.distanceTransform((~edge_a).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_b = cv2.distanceTransform((~edge_b).astype(np.uint8), cv2.DIST_L2, 5)
    band = (dist_to_a < max_gap_px) & (dist_to_b < max_gap_px) & (a.mask | b.mask)
    band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if int(band.sum()) < 400:
        union_mask = a.mask | b.mask
        return union_mask, {"mode": "union_fallback", "band_area_px": int(band.sum())}
    return band, {"mode": "joint_band", "band_area_px": int(band.sum())}


def _select_pair_from_summary(
    candidates: List[SegmentCandidate],
    candidate_pairs: List[Dict[str, object]],
    preferred_ids: Optional[Sequence[str]] = None,
) -> List[SegmentCandidate]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    preferred = {item for item in (preferred_ids or []) if item}
    for item in candidate_pairs:
        pair_ids = [str(v).strip().upper() for v in item.get("pair", [])]
        if len(pair_ids) != 2:
            continue
        if preferred and not preferred.issubset(set(pair_ids)):
            continue
        pair_score = float(item.get("pair_score", 0.0))
        near_area = int(item.get("near_band_area_px", 0))
        band_linearity = float(item.get("band_linearity", 0.0))
        gap = item.get("min_edge_gap_px")
        gap_ok = gap is None or float(gap) <= 12.0
        if pair_score < 0.78 or near_area < 2000 or not gap_ok:
            continue
        selected: List[SegmentCandidate] = []
        for cid in pair_ids:
            candidate = by_id.get(cid)
            if candidate is None:
                selected = []
                break
            if candidate.border_touch_ratio > 0.28:
                selected = []
                break
            if candidate.image_area_ratio > 0.38:
                selected = []
                break
            selected.append(candidate)
        if len(selected) == 2:
            if band_linearity < 0.08 and pair_score < 0.9:
                continue
            return selected
    return []


class QwenReasoner:
    def __init__(self, python_executable: str, model_path: str, max_new_tokens: int) -> None:
        self.python_executable = python_executable
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.helper_script = os.path.join(os.path.dirname(__file__), "qwen_infer.py")

    def is_available(self) -> bool:
        return os.path.exists(self.python_executable) and os.path.exists(self.helper_script) and os.path.exists(self.model_path)

    def select(
        self,
        color: np.ndarray,
        candidates: List[SegmentCandidate],
        candidate_pairs: List[Dict[str, object]],
        work_dir: str,
        frame_id: str,
        prompt: str,
    ) -> Optional[Dict[str, object]]:
        if not self.is_available() or not candidates:
            return None
        ensure_dir(work_dir)
        raw_image_path = os.path.join(work_dir, f"{frame_id}_qwen_color.png")
        overlay_image_path = os.path.join(work_dir, f"{frame_id}_qwen_candidates.png")
        candidates_json_path = os.path.join(work_dir, f"{frame_id}_qwen_candidates.json")
        output_json_path = os.path.join(work_dir, f"{frame_id}_qwen_reasoning.json")
        overlay = render_candidate_overlay(color, candidates)
        imageio.imwrite(raw_image_path, color)
        imageio.imwrite(overlay_image_path, overlay)
        payload = {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "candidate_pairs": candidate_pairs,
            "prompt": prompt,
        }
        with open(candidates_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        cmd = [
            self.python_executable,
            self.helper_script,
            "--raw_image_path",
            raw_image_path,
            "--overlay_image_path",
            overlay_image_path,
            "--candidates_json",
            candidates_json_path,
            "--output_json",
            output_json_path,
            "--model_path",
            self.model_path,
            "--max_new_tokens",
            str(self.max_new_tokens),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(output_json_path):
            return {
                "backend": "qwen_local",
                "status": "failed",
                "stderr": proc.stderr.strip(),
                "stdout": proc.stdout.strip(),
                "raw_image_path": raw_image_path,
                "overlay_path": overlay_image_path,
            }
        out = json.load(open(output_json_path, "r", encoding="utf-8"))
        out["raw_image_path"] = raw_image_path
        out["overlay_path"] = overlay_image_path
        return out


class GLMReasoner:
    def __init__(self, python_executable: str, model_name: str, api_key_env: str, temperature: float) -> None:
        self.python_executable = python_executable
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.helper_script = os.path.join(os.path.dirname(__file__), "glm_infer.py")

    def is_available(self) -> bool:
        return os.path.exists(self.python_executable) and os.path.exists(self.helper_script) and bool(os.environ.get(self.api_key_env) or os.environ.get("BIGMODEL_API_KEY"))

    def select(
        self,
        color: np.ndarray,
        candidates: List[SegmentCandidate],
        candidate_pairs: List[Dict[str, object]],
        work_dir: str,
        frame_id: str,
        prompt: str,
    ) -> Optional[Dict[str, object]]:
        if not self.is_available() or not candidates:
            return None
        ensure_dir(work_dir)
        raw_image_path = os.path.join(work_dir, f"{frame_id}_glm_color.png")
        overlay_image_path = os.path.join(work_dir, f"{frame_id}_glm_candidates.png")
        candidates_json_path = os.path.join(work_dir, f"{frame_id}_glm_candidates.json")
        output_json_path = os.path.join(work_dir, f"{frame_id}_glm_reasoning.json")
        overlay = render_candidate_overlay(color, candidates)
        imageio.imwrite(raw_image_path, color)
        imageio.imwrite(overlay_image_path, overlay)
        payload = {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "candidate_pairs": candidate_pairs,
            "prompt": prompt,
        }
        with open(candidates_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        cmd = [
            self.python_executable,
            self.helper_script,
            "--raw_image_path",
            raw_image_path,
            "--overlay_image_path",
            overlay_image_path,
            "--candidates_json",
            candidates_json_path,
            "--output_json",
            output_json_path,
            "--model_name",
            self.model_name,
            "--api_key_env",
            self.api_key_env,
            "--temperature",
            str(self.temperature),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(output_json_path):
            return {
                "backend": "glm_online",
                "status": "failed",
                "stderr": proc.stderr.strip(),
                "stdout": proc.stdout.strip(),
                "raw_image_path": raw_image_path,
                "overlay_path": overlay_image_path,
            }
        out = json.load(open(output_json_path, "r", encoding="utf-8"))
        out["raw_image_path"] = raw_image_path
        out["overlay_path"] = overlay_image_path
        return out


def resolve_reasoned_candidates(
    candidates: List[SegmentCandidate],
    reasoning: Optional[Dict[str, object]],
    candidate_pairs: Optional[List[Dict[str, object]]] = None,
    top_k: int = 2,
) -> List[SegmentCandidate]:
    if not candidates:
        return []
    if reasoning is None:
        pair_selected = _select_pair_from_summary(candidates, candidate_pairs or [])
        return pair_selected[:top_k] if pair_selected else select_geometry_candidates(candidates, top_k)
    plate_ids = []
    roi_ids = []
    selected_ids = []
    for key, bucket in (
        ("plate_candidate_ids", plate_ids),
        ("roi_candidate_ids", roi_ids),
        ("selected_candidate_ids", selected_ids),
    ):
        ids = reasoning.get(key)
        if isinstance(ids, list):
            bucket.extend(str(item).strip().upper() for item in ids)
    plate_ids = [item for item in plate_ids if item]
    roi_ids = [item for item in roi_ids if item]
    selected_ids = [item for item in selected_ids if item]
    if len(plate_ids) >= 2:
        wanted = plate_ids
    elif roi_ids:
        wanted = roi_ids
    else:
        wanted = selected_ids
    if not wanted:
        pair_selected = _select_pair_from_summary(candidates, candidate_pairs or [])
        return pair_selected[:top_k] if pair_selected else select_geometry_candidates(candidates, top_k)
    selected = [candidate for candidate in candidates if candidate.candidate_id in wanted]
    selected = [candidate for candidate in selected if candidate.border_touch_ratio <= 0.22]
    selected.sort(key=lambda candidate: wanted.index(candidate.candidate_id) if candidate.candidate_id in wanted else 999)
    if len(selected) >= top_k:
        return selected[:top_k]
    pair_selected = _select_pair_from_summary(candidates, candidate_pairs or [], preferred_ids=plate_ids[:2] if len(plate_ids) >= 2 else None)
    if pair_selected:
        return pair_selected[:top_k]
    return selected[:top_k] if selected else select_geometry_candidates(candidates, top_k)


def serialize_reasoning_for_json(reasoning: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if reasoning is None:
        return None
    cleaned: Dict[str, object] = {}
    for key, value in reasoning.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = value
        elif isinstance(value, dict):
            cleaned[key] = value
    return cleaned


def extract_json_object(text: str) -> Optional[Dict[str, object]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
