from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


def depth_to_xyz(depth_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    valid = np.isfinite(depth_m) & (depth_m > 0)
    z = np.where(valid, depth_m, 0.0).astype(np.float32)
    x = np.zeros_like(z, dtype=np.float32)
    y = np.zeros_like(z, dtype=np.float32)
    x[valid] = ((xs[valid] - k[0, 2]) * z[valid] / k[0, 0]).astype(np.float32)
    y[valid] = ((ys[valid] - k[1, 2]) * z[valid] / k[1, 1]).astype(np.float32)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def estimate_normals_from_depth(depth_m: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = depth_to_xyz(depth_m, k)
    zx = cv2.Sobel(xyz[:, :, 2], cv2.CV_32F, 1, 0, ksize=3)
    zy = cv2.Sobel(xyz[:, :, 2], cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.Sobel(xyz[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    xy = cv2.Sobel(xyz[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    yx = cv2.Sobel(xyz[:, :, 1], cv2.CV_32F, 1, 0, ksize=3)
    yy = cv2.Sobel(xyz[:, :, 1], cv2.CV_32F, 0, 1, ksize=3)

    dx = np.stack([xx, yx, zx], axis=-1)
    dy = np.stack([xy, yy, zy], axis=-1)
    normals = np.cross(dx, dy)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(depth_m) & (depth_m > 0) & (norms[:, :, 0] > 1e-6)
    normals = np.divide(normals, np.maximum(norms, 1e-6), out=np.zeros_like(normals), where=norms > 1e-6)
    normals[:, :, 2] = np.where(valid, np.abs(normals[:, :, 2]), 0.0)

    mean_normal = cv2.blur(normals, (11, 11))
    mean_norm = np.linalg.norm(mean_normal, axis=-1, keepdims=True)
    mean_normal = np.divide(mean_normal, np.maximum(mean_norm, 1e-6), out=np.zeros_like(mean_normal), where=mean_norm > 1e-6)
    cosine = np.sum(normals * mean_normal, axis=-1)
    normal_change = 1.0 - np.clip(cosine, -1.0, 1.0)
    normal_change[~valid] = 0.0
    return normals.astype(np.float32), normal_change.astype(np.float32)


def fit_plane_ransac(points: np.ndarray, threshold: float, min_points: int) -> Optional[Dict[str, np.ndarray]]:
    if len(points) < min_points:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    model, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=1200)
    if len(inliers) < min_points:
        return None
    normal = np.asarray(model[:3], dtype=np.float32)
    normal /= max(np.linalg.norm(normal), 1e-6)
    plane_points = points[np.asarray(inliers)]
    centroid = plane_points.mean(axis=0).astype(np.float32)
    return {
        "model": np.asarray(model, dtype=np.float32),
        "normal": normal,
        "centroid": centroid,
        "points": plane_points.astype(np.float32),
        "inliers": np.asarray(inliers, dtype=np.int32),
    }


def extract_two_planes(xyz_map: np.ndarray, valid_mask: np.ndarray, threshold: float, min_points: int) -> List[Dict[str, np.ndarray]]:
    flat_xyz = xyz_map.reshape(-1, 3)
    flat_valid = valid_mask.reshape(-1)
    available_idx = np.where(flat_valid)[0]
    available_points = flat_xyz[available_idx]
    planes: List[Dict[str, np.ndarray]] = []
    for _ in range(2):
        plane = fit_plane_ransac(available_points, threshold=threshold, min_points=min_points)
        if plane is None:
            break
        global_inliers = available_idx[plane["inliers"]]
        mask = np.zeros(flat_valid.shape[0], dtype=bool)
        mask[global_inliers] = True
        planes.append(
            {
                "model": plane["model"],
                "normal": plane["normal"],
                "centroid": plane["centroid"],
                "mask": mask.reshape(xyz_map.shape[:2]),
                "points": plane["points"],
            }
        )
        keep = np.ones(len(available_points), dtype=bool)
        keep[plane["inliers"]] = False
        available_points = available_points[keep]
        available_idx = available_idx[keep]
        if len(available_points) < min_points:
            break
    return planes


def largest_component(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    labels, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[1:3]
    if stats.shape[0] <= 1:
        return mask.astype(bool)
    best_id = 0
    best_area = 0
    for idx in range(1, stats.shape[0]):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > best_area and area >= min_area:
            best_area = area
            best_id = idx
    return labels == best_id


def normalize_map(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = arr[valid]
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if hi <= lo + 1e-6:
        out[valid] = 1.0
        return out
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def estimate_surface_curvature(xyz_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    curvature = np.zeros(xyz_map.shape[:2], dtype=np.float32)
    if not np.any(valid_mask):
        return curvature
    for channel in range(3):
        plane = xyz_map[:, :, channel].astype(np.float32)
        lap = cv2.Laplacian(plane, cv2.CV_32F, ksize=3)
        curvature += np.abs(lap)
    curvature[~valid_mask] = 0.0
    return curvature


def ordered_centerline(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    order = np.argsort(centered @ axis)
    pts = pts[order]
    chunks = np.array_split(pts, min(80, max(20, len(pts) // 10)))
    line = [chunk.mean(axis=0) for chunk in chunks if len(chunk) > 0]
    if len(line) < 2:
        return None
    return np.asarray(line, dtype=np.float32)


def _principal_axis_from_mask(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    ortho = np.array([-axis[1], axis[0]], dtype=np.float32)
    proj = centered @ axis
    return pts, mean, axis.astype(np.float32), proj.astype(np.float32)


def weighted_centerline_from_score(
    score_map: np.ndarray,
    support_mask: np.ndarray,
    min_points: int = 24,
) -> Optional[np.ndarray]:
    support = support_mask & np.isfinite(score_map) & (score_map > 0)
    axis_fit = _principal_axis_from_mask(support)
    if axis_fit is None:
        return None
    pts, mean, axis, proj = axis_fit
    span = float(proj.max() - proj.min())
    if span < 20.0:
        return None

    bin_count = int(np.clip(span / 6.0, 18, 80))
    edges = np.linspace(float(proj.min()), float(proj.max()), bin_count + 1)
    line: List[np.ndarray] = []
    centered = pts - mean
    scores = score_map[pts[:, 1].astype(np.int32), pts[:, 0].astype(np.int32)].astype(np.float32)
    for idx in range(bin_count):
        if idx == bin_count - 1:
            in_bin = (proj >= edges[idx]) & (proj <= edges[idx + 1])
        else:
            in_bin = (proj >= edges[idx]) & (proj < edges[idx + 1])
        if int(in_bin.sum()) < 5:
            continue
        bin_scores = scores[in_bin]
        keep = bin_scores >= np.percentile(bin_scores, 70)
        if int(keep.sum()) < 3:
            continue
        bin_pts = pts[in_bin][keep]
        bin_weights = np.clip(bin_scores[keep], 1e-4, None)
        centroid = np.average(bin_pts, axis=0, weights=bin_weights)
        line.append(centroid.astype(np.float32))
    if len(line) < min_points:
        return None
    return np.asarray(line, dtype=np.float32)


def groove_mask_from_centerline(points_uv: np.ndarray, shape: Tuple[int, int], thickness: int = 9) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if points_uv is None or len(points_uv) < 2:
        return mask.astype(bool)
    pts = np.round(points_uv).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
    cv2.polylines(mask, [pts.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask > 0


def smooth_centerline(points_uv: np.ndarray, window: int = 7) -> np.ndarray:
    if points_uv is None or len(points_uv) < 5:
        return points_uv
    pts = points_uv.astype(np.float32)
    radius = max(window // 2, 1)
    padded = np.pad(pts, ((radius, radius), (0, 0)), mode='edge')
    smoothed = []
    for idx in range(len(pts)):
        chunk = padded[idx: idx + 2 * radius + 1]
        smoothed.append(chunk.mean(axis=0))
    return np.asarray(smoothed, dtype=np.float32)


def refine_centerline_with_3d(
    centerline_px: np.ndarray,
    refine_score: np.ndarray,
    support_mask: np.ndarray,
    search_radius: int = 8,
) -> np.ndarray:
    if centerline_px is None or len(centerline_px) < 3:
        return centerline_px
    pts = centerline_px.astype(np.float32).copy()
    refined: List[np.ndarray] = []
    h, w = refine_score.shape
    for idx in range(len(pts)):
        prev_pt = pts[max(idx - 1, 0)]
        next_pt = pts[min(idx + 1, len(pts) - 1)]
        tangent = next_pt - prev_pt
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent = tangent / tangent_norm
        ortho = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        center = pts[idx]
        best_uv = center
        cx = int(round(float(center[0])))
        cy = int(round(float(center[1])))
        if 0 <= cx < w and 0 <= cy < h and support_mask[cy, cx]:
            best_score = float(refine_score[cy, cx])
        else:
            best_score = -1.0
        for step in range(-search_radius, search_radius + 1):
            probe = center + ortho * float(step)
            u = int(round(float(probe[0])))
            v = int(round(float(probe[1])))
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            if not support_mask[v, u]:
                continue
            score = float(refine_score[v, u])
            if score > best_score:
                best_score = score
                best_uv = np.array([u, v], dtype=np.float32)
        refined.append(best_uv)
    return np.asarray(refined, dtype=np.float32)


def centerline_straightness(points_uv: np.ndarray) -> float:
    if points_uv is None or len(points_uv) < 3:
        return 0.0
    pts = points_uv.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    ortho = np.stack([-axis[1], axis[0]], axis=0)
    residual = np.abs(centered @ ortho)
    span = float((centered @ axis).max() - (centered @ axis).min())
    if span <= 1e-6:
        return 0.0
    return float(1.0 - np.clip(np.median(residual) / max(span * 0.08, 1e-6), 0.0, 1.0))


def centerline_smoothness(points_uv: np.ndarray) -> float:
    if points_uv is None or len(points_uv) < 5:
        return 0.0
    pts = points_uv.astype(np.float32)
    v0 = pts[1:-1] - pts[:-2]
    v1 = pts[2:] - pts[1:-1]
    n0 = np.linalg.norm(v0, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    valid = (n0 > 1e-6) & (n1 > 1e-6)
    if not np.any(valid):
        return 0.0
    cosines = np.sum(v0[valid] * v1[valid], axis=1) / np.maximum(n0[valid] * n1[valid], 1e-6)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    mean_angle = float(np.mean(angles))
    return float(1.0 - np.clip(mean_angle / 0.7, 0.0, 1.0))


def semantic_hint_from_prompt(prompt: str) -> Dict[str, float]:
    text = prompt.lower()
    return {
        "prefer_dark_line": 1.0 if any(k in text for k in ["v", "groove", "joint", "seam", "weld", "坡口", "焊缝", "拼接"]) else 0.7,
        "prefer_normal_change": 1.0 if any(k in text for k in ["v", "坡口", "groove"]) else 0.6,
    }


def _fit_line_outputs(points_3d: np.ndarray, uv: np.ndarray, approach: np.ndarray) -> Dict[str, np.ndarray]:
    center = points_3d.mean(axis=0)
    centered = points_3d - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    seam_dir = vt[0]
    if seam_dir[2] > 0:
        seam_dir = -seam_dir
    proj = centered @ seam_dir
    start = center + seam_dir * proj.min()
    end = center + seam_dir * proj.max()
    return {
        "groove_mask": None,
        "centerline_pixels": uv.astype(np.float32),
        "centerline_points_m": points_3d.astype(np.float32),
        "target_point_m": center.astype(np.float32),
        "seam_direction_camera": seam_dir.astype(np.float32),
        "approach_direction_camera": approach.astype(np.float32),
        "weld_start_point_m": start.astype(np.float32),
        "weld_end_point_m": end.astype(np.float32),
    }


def extract_groove_from_roi(
    color: np.ndarray,
    depth_m: np.ndarray,
    xyz_map: np.ndarray,
    k_depth: np.ndarray,
    roi_mask: np.ndarray,
    prompt: str,
    groove_score_threshold: float,
    min_groove_area_px: int,
    semantic_color_weight: float,
    geometry_normal_weight: float,
    geometry_depth_weight: float,
    roi_center_prior_weight: float,
) -> Optional[Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    valid = np.isfinite(depth_m) & (depth_m > 0) & roi_mask
    if valid.sum() < max(min_groove_area_px * 5, 500):
        return None

    normals, normal_change = estimate_normals_from_depth(depth_m, k_depth)
    curvature = estimate_surface_curvature(xyz_map, valid)
    depth_fill = depth_m.copy().astype(np.float32)
    depth_fill[~np.isfinite(depth_fill)] = float(np.median(depth_fill[valid]))
    blur = cv2.GaussianBlur(depth_fill, (0, 0), 1.2)
    local = cv2.GaussianBlur(depth_fill, (0, 0), 6.5)
    valley = blur - local
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    edge = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    low_sat = 1.0 - normalize_map(hsv[:, :, 1].astype(np.float32), valid)

    hint = semantic_hint_from_prompt(prompt)
    semantic_score = 0.55 * normalize_map(blackhat.astype(np.float32), valid) + 0.25 * edge + 0.20 * low_sat
    geometry_score = (
        hint["prefer_normal_change"] * normalize_map(normal_change, valid) +
        hint["prefer_dark_line"] * normalize_map(valley, valid)
    ) / max(hint["prefer_normal_change"] + hint["prefer_dark_line"], 1e-6)
    score = (
        semantic_color_weight * semantic_score +
        geometry_normal_weight * normalize_map(normal_change, valid) +
        geometry_depth_weight * normalize_map(valley, valid)
    )
    score = 0.65 * score + 0.35 * geometry_score
    roi_dist = cv2.distanceTransform(roi_mask.astype(np.uint8), cv2.DIST_L2, 5)
    roi_center_prior = normalize_map(roi_dist.astype(np.float32), roi_mask)
    score = score * ((1.0 - roi_center_prior_weight) + roi_center_prior_weight * roi_center_prior)
    score[~valid] = 0.0

    thick_thr = max(groove_score_threshold * 0.85, float(np.percentile(score[valid], 88)))
    thick_band_mask = score > thick_thr
    thick_band_mask = cv2.morphologyEx(thick_band_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2).astype(bool)
    thick_band_mask = cv2.morphologyEx(thick_band_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    thick_band_mask = largest_component(thick_band_mask, min_area=max(min_groove_area_px * 2, 240))
    thick_band_mask &= roi_mask

    curvature_norm = normalize_map(curvature, valid)
    valley_norm = normalize_map(valley, valid)
    normal_norm = normalize_map(normal_change, valid)
    seam_refine_score = 0.45 * normal_norm + 0.30 * curvature_norm + 0.25 * valley_norm
    seam_refine_score *= (0.55 + 0.45 * roi_center_prior)
    seam_refine_score[~(valid & thick_band_mask)] = 0.0

    thin_valid = valid & thick_band_mask
    if int(thin_valid.sum()) < max(min_groove_area_px, 120):
        return None
    thin_thr = max(0.35, float(np.percentile(seam_refine_score[thin_valid], 90)))
    thin_seam_mask = seam_refine_score > thin_thr
    thin_seam_mask = cv2.morphologyEx(thin_seam_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    thin_seam_mask = cv2.morphologyEx(thin_seam_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    thin_seam_mask = largest_component(thin_seam_mask, min_area=min_groove_area_px)
    thin_seam_mask &= thick_band_mask

    center_support = valid & thick_band_mask & (roi_center_prior > 0.25)
    weighted_centerline = weighted_centerline_from_score(seam_refine_score, center_support, min_points=18)
    threshold_centerline = ordered_centerline(thin_seam_mask)
    if weighted_centerline is not None:
        centerline_px = weighted_centerline
    else:
        centerline_px = threshold_centerline
    if centerline_px is None:
        return None
    centerline_px = refine_centerline_with_3d(centerline_px, seam_refine_score, thick_band_mask, search_radius=9)
    centerline_px = smooth_centerline(centerline_px, window=7)
    thin_seam_mask = groove_mask_from_centerline(centerline_px, roi_mask.shape, thickness=5) & thick_band_mask
    groove_mask = groove_mask_from_centerline(centerline_px, roi_mask.shape, thickness=9) & thick_band_mask
    straightness = centerline_straightness(centerline_px)
    smoothness = centerline_smoothness(centerline_px)
    if straightness > 0.9 and smoothness < 0.5 and len(centerline_px) >= 10:
        alpha = np.linspace(0.0, 1.0, len(centerline_px), dtype=np.float32)[:, None]
        line_fit = centerline_px[0:1] * (1.0 - alpha) + centerline_px[-1:] * alpha
        centerline_px = 0.5 * centerline_px + 0.5 * line_fit
        centerline_px = smooth_centerline(centerline_px, window=9)
        straightness = centerline_straightness(centerline_px)
        smoothness = centerline_smoothness(centerline_px)
        groove_mask = groove_mask_from_centerline(centerline_px, roi_mask.shape, thickness=9) & roi_mask
    if straightness < 0.35 and smoothness < 0.45:
        return None

    uv = np.round(centerline_px).astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, xyz_map.shape[1] - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, xyz_map.shape[0] - 1)
    points_3d = xyz_map[uv[:, 1], uv[:, 0]]
    valid_points = np.isfinite(points_3d[:, 2]) & (points_3d[:, 2] > 0)
    uv = uv[valid_points]
    points_3d = points_3d[valid_points]
    if len(points_3d) < 8:
        return None

    approach = -points_3d.mean(axis=0)
    approach = approach / max(np.linalg.norm(approach), 1e-6)
    out = _fit_line_outputs(points_3d, uv, approach)
    out["groove_mask"] = groove_mask
    out["thick_band_mask"] = thick_band_mask
    out["thin_seam_mask"] = thin_seam_mask
    out["normals"] = normals
    out["normal_change"] = normal_change
    out["curvature_map"] = curvature
    out["score_map"] = seam_refine_score
    out["coarse_score_map"] = score
    out["roi_center_prior"] = roi_center_prior
    out["centerline_straightness"] = straightness
    out["centerline_smoothness"] = smoothness
    return out


def generate_waypoints(target_point: np.ndarray, approach_dir: np.ndarray, start: np.ndarray, end: np.ndarray, pre_offset: float, approach_offset: float) -> Dict[str, np.ndarray]:
    approach_dir = approach_dir / max(np.linalg.norm(approach_dir), 1e-6)
    return {
        "pre_approach_point_m": target_point - approach_dir * pre_offset,
        "approach_point_m": target_point - approach_dir * approach_offset,
        "target_point_m": target_point,
        "weld_start_point_m": start,
        "weld_end_point_m": end,
    }
