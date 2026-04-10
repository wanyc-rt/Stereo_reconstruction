import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d


def build_k(intrinsic: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_transform_matrix(rot_flat: List[float], trans_mm: List[float], translation_scale: float = 0.001) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(rot_flat, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.array(trans_mm, dtype=np.float32) * translation_scale
    return transform


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_stems(folder: str, suffixes: Tuple[str, ...] = (".png", ".npy")) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith(suffixes):
            out[os.path.splitext(name)[0]] = os.path.join(folder, name)
    return out


def list_foundation_depths(folder: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith('_pred_depth_m.npy'):
            stem = name[:-len('_pred_depth_m.npy')]
            out[stem] = os.path.join(folder, name)
    return out


def load_color(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[..., :3].astype(np.uint8)


def load_depth(path: str, depth_scale: float = 0.001) -> np.ndarray:
    if path.lower().endswith('.npy'):
        depth = np.load(path).astype(np.float32)
    else:
        depth = imageio.imread(path).astype(np.float32)
        depth *= depth_scale
    return depth


def make_depth_vis(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth_m[valid]
        lo = float(np.percentile(vals, 5))
        hi = float(np.percentile(vals, 95))
        scaled = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        vis = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)[:, :, ::-1]


def make_normal_vis(normals: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    vis = ((np.clip(normals, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    if valid_mask is not None:
        vis = vis.copy()
        vis[~valid_mask] = 0
    return vis


def copy_file_if_exists(src: Optional[str], dst: str) -> bool:
    if src is None or not os.path.exists(src):
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def load_camera_meta(data_dir: str) -> Tuple[dict, np.ndarray]:
    meta_path = os.path.join(data_dir, 'camera_intrinsics.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    depth_intrinsic = meta['streams']['Depth']['intrinsic']
    return meta, build_k(depth_intrinsic)


def extract_camera_alignment(meta: dict) -> Dict[str, Optional[np.ndarray]]:
    out: Dict[str, Optional[np.ndarray]] = {
        'k_depth': None,
        'k_color': None,
        't_depth_to_color': None,
    }
    try:
        out['k_depth'] = build_k(meta['streams']['Depth']['intrinsic'])
    except Exception:
        pass
    try:
        out['k_color'] = build_k(meta['streams']['Color']['intrinsic'])
    except Exception:
        pass
    try:
        transform_meta = meta['pipeline_camera_param']['transform']
        out['t_depth_to_color'] = build_transform_matrix(transform_meta['rot'], transform_meta['transform'], translation_scale=0.001)
    except Exception:
        pass
    return out


def nearest_stem(target: str, candidates: Dict[str, str], max_distance: Optional[int] = None) -> Optional[str]:
    if not candidates:
        return None
    if target in candidates:
        return target
    try:
        target_num = int(target)
        numeric = sorted((int(k), k) for k in candidates.keys() if k.isdigit())
        if not numeric:
            return next(iter(candidates.keys()))
        best = min(numeric, key=lambda item: abs(item[0] - target_num))
        if max_distance is not None and abs(best[0] - target_num) > max_distance:
            return None
        return best[1]
    except Exception:
        return next(iter(candidates.keys()))


def resolve_foundation_depth_dir(data_dir: str, foundation_depth_dir: Optional[str] = None) -> str:
    if foundation_depth_dir:
        return foundation_depth_dir
    return os.path.join(data_dir, 'output_stride_1', 'depth')


def collect_frames(
    data_dir: str,
    frame_id: Optional[str] = None,
    start_frame_id: Optional[str] = None,
    max_frames: int = -1,
    depth_source: str = 'foundation_stereo',
    foundation_depth_dir: Optional[str] = None,
    max_color_frame_delta: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    color_files = list_stems(os.path.join(data_dir, 'color'), ('.png', '.jpg', '.jpeg'))
    raw_depth_files = list_stems(os.path.join(data_dir, 'depth'), ('.npy', '.png'))
    fs_depth_dir = resolve_foundation_depth_dir(data_dir, foundation_depth_dir)
    fs_depth_files = list_foundation_depths(fs_depth_dir)
    foundation_root = os.path.dirname(fs_depth_dir)

    if depth_source == 'foundation_stereo':
        depth_files = fs_depth_files
        depth_kind = 'foundation_stereo'
    elif depth_source == 'raw':
        depth_files = raw_depth_files
        depth_kind = 'raw'
    else:
        depth_files = fs_depth_files if fs_depth_files else raw_depth_files
        depth_kind = 'foundation_stereo' if fs_depth_files else 'raw'

    depth_stems = sorted(depth_files.keys())
    if frame_id is not None:
        depth_stems = [stem for stem in depth_stems if stem == frame_id]
    elif start_frame_id is not None:
        depth_stems = [stem for stem in depth_stems if stem >= start_frame_id]
    if max_frames > 0:
        depth_stems = depth_stems[:max_frames]

    frames: List[Dict[str, Optional[str]]] = []
    for stem in depth_stems:
        color_stem = nearest_stem(stem, color_files, max_distance=max_color_frame_delta)
        frames.append(
            {
                'frame_id': stem,
                'color_frame_id': color_stem,
                'color_path': color_files.get(color_stem) if color_stem is not None else None,
                'depth_path': depth_files[stem],
                'depth_source': depth_kind,
                'foundation_root': foundation_root if depth_kind == 'foundation_stereo' else None,
            }
        )
    return frames


def save_json(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    out[mask] = (0.65 * out[mask] + 0.35 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return out


def draw_polyline(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = image.copy()
    if len(points) < 2:
        return out
    points = np.round(points).astype(np.int32)
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(out, tuple(p0.tolist()), tuple(p1.tolist()), color, thickness, cv2.LINE_AA)
    return out


def project_points_to_color(
    xyz_depth: np.ndarray,
    color_img: np.ndarray,
    k_color: np.ndarray,
    t_depth_to_color: np.ndarray,
) -> np.ndarray:
    points = xyz_depth.reshape(-1, 3)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    valid_depth = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    if not np.any(valid_depth):
        return colors
    points_h = np.concatenate([points[valid_depth], np.ones((int(valid_depth.sum()), 1), dtype=np.float32)], axis=1)
    points_color = (t_depth_to_color @ points_h.T).T[:, :3]
    valid_color = points_color[:, 2] > 1e-6
    if not np.any(valid_color):
        return colors
    points_color_valid = points_color[valid_color]
    us = np.round(points_color_valid[:, 0] * k_color[0, 0] / points_color_valid[:, 2] + k_color[0, 2]).astype(np.int32)
    vs = np.round(points_color_valid[:, 1] * k_color[1, 1] / points_color_valid[:, 2] + k_color[1, 2]).astype(np.int32)
    inside = (us >= 0) & (us < color_img.shape[1]) & (vs >= 0) & (vs < color_img.shape[0])
    if not np.any(inside):
        return colors
    valid_depth_ids = np.where(valid_depth)[0]
    valid_color_ids = valid_depth_ids[valid_color]
    sample_ids = valid_color_ids[inside]
    colors[sample_ids] = color_img[vs[inside], us[inside], :3]
    return colors


def render_color_aligned_to_depth(
    xyz_depth: np.ndarray,
    color_img: np.ndarray,
    k_color: np.ndarray,
    t_depth_to_color: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = xyz_depth.shape[:2]
    aligned = np.zeros((h, w, 3), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=bool)
    colors_flat = project_points_to_color(xyz_depth, color_img, k_color, t_depth_to_color)
    points = xyz_depth.reshape(-1, 3)
    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.reshape(-1)
    color_valid = np.any(colors_flat > 0, axis=1)
    keep = valid & color_valid
    if not np.any(keep):
        return aligned, coverage
    aligned.reshape(-1, 3)[keep] = colors_flat[keep]
    coverage.reshape(-1)[keep] = True
    return aligned, coverage


def make_alignment_overlay(
    color_depth_aligned: np.ndarray,
    depth_m: np.ndarray,
    coverage_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    depth_vis = make_depth_vis(depth_m)
    gray = cv2.cvtColor(color_depth_aligned, cv2.COLOR_RGB2GRAY)
    color_edges = cv2.Canny(gray, 50, 150)
    depth_u8 = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_edges = cv2.Canny(depth_u8, 30, 90)
    overlay = depth_vis.copy()
    overlay[depth_edges > 0] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[color_edges > 0] = np.array([0, 255, 0], dtype=np.uint8)
    if coverage_mask is not None:
        missing = ~coverage_mask
        overlay[missing] = (0.65 * overlay[missing]).astype(np.uint8)
    return overlay


def save_colored_point_cloud(xyz_map: np.ndarray, colors_flat: np.ndarray, out_path: str, valid_mask: Optional[np.ndarray] = None) -> None:
    points = xyz_map.reshape(-1, 3)
    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.reshape(-1)
    pts = points[valid].astype(np.float64)
    cols = colors_flat.reshape(-1, 3)[valid].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(out_path, pcd)
