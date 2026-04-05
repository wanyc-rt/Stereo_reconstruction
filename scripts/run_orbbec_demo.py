import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from Utils import depth2xyzmap, set_logging_format, set_seed, toOpen3dCloud, vis_disparity
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

DEVICE_PRESETS = {
    "gemini_435le": {
        "display_name": "Gemini 435Le",
        "stereo_baseline_m": 0.095,
        "depth_scale_m": 0.001,
        "notes": "Depth values are treated as millimeters and converted to meters. Stereo baseline defaults to 95 mm.",
    }
}


def build_k(intrinsic: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def list_pngs(folder: str) -> Dict[str, str]:
    files = {}
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".png"):
            files[os.path.splitext(name)[0]] = os.path.join(folder, name)
    return files


def ensure_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def load_gray_as_rgb(path: str) -> np.ndarray:
    img = imageio.imread(path)
    return ensure_three_channel(img)


def load_color_rgb(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = ensure_three_channel(img)
    return img[..., :3]


def depth_to_meters(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    if depth_scale > 0:
        depth = depth * depth_scale
    return depth


def normalize_device_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def get_device_preset(device_name: str) -> Optional[Dict[str, float]]:
    return DEVICE_PRESETS.get(normalize_device_name(device_name))


def make_transform_matrix(rot_flat: List[float], trans_mm: List[float], translation_scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(rot_flat, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.array(trans_mm, dtype=np.float32) * translation_scale
    return transform


def project_points_to_color(
    xyz_depth: np.ndarray,
    color_img: np.ndarray,
    k_color: np.ndarray,
    t_depth_to_color: np.ndarray,
) -> np.ndarray:
    points = xyz_depth.reshape(-1, 3)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    valid_depth = points[:, 2] > 0
    if not np.any(valid_depth):
        return colors

    points_h = np.concatenate(
        [points[valid_depth], np.ones((valid_depth.sum(), 1), dtype=np.float32)], axis=1
    )
    points_color = (t_depth_to_color @ points_h.T).T[:, :3]
    valid_color = points_color[:, 2] > 1e-6
    if not np.any(valid_color):
        return colors

    points_color_valid = points_color[valid_color]
    us = np.round(
        points_color_valid[:, 0] * k_color[0, 0] / points_color_valid[:, 2] + k_color[0, 2]
    ).astype(np.int32)
    vs = np.round(
        points_color_valid[:, 1] * k_color[1, 1] / points_color_valid[:, 2] + k_color[1, 2]
    ).astype(np.int32)

    inside = (
        (us >= 0)
        & (us < color_img.shape[1])
        & (vs >= 0)
        & (vs < color_img.shape[0])
    )
    if not np.any(inside):
        return colors

    valid_depth_ids = np.where(valid_depth)[0]
    valid_color_ids = valid_depth_ids[valid_color]
    sample_ids = valid_color_ids[inside]
    colors[sample_ids] = color_img[vs[inside], us[inside], :3]
    return colors


def compute_metrics(pred_depth_m: np.ndarray, gt_depth_m: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = pred_depth_m[mask]
    gt = gt_depth_m[mask]
    abs_err = np.abs(pred - gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    abs_rel = np.mean(abs_err / np.clip(gt, 1e-6, None))
    ratio = np.maximum(pred / np.clip(gt, 1e-6, None), gt / np.clip(pred, 1e-6, None))
    delta1 = np.mean(ratio < 1.25)
    return {
        "num_valid": int(mask.sum()),
        "mae_m": float(abs_err.mean()),
        "rmse_m": float(rmse),
        "abs_rel": float(abs_rel),
        "delta1": float(delta1),
        "pred_mean_m": float(pred.mean()),
        "gt_mean_m": float(gt.mean()),
    }


def compute_depth_stats(depth_m: np.ndarray, z_far: float) -> Dict[str, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < z_far)
    out = {"valid_ratio": float(valid.mean())}
    if valid.any():
        vals = depth_m[valid]
        out.update(
            {
                "mean_m": float(vals.mean()),
                "median_m": float(np.median(vals)),
                "min_m": float(vals.min()),
                "max_m": float(vals.max()),
            }
        )
    return out


def compute_classic_disparities(left_gray: np.ndarray, right_gray: np.ndarray) -> Dict[str, np.ndarray]:
    num_disp = 128
    block_size = 9
    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    bm.setTextureThreshold(10)
    bm.setUniquenessRatio(10)
    bm.setSpeckleWindowSize(100)
    bm.setSpeckleRange(16)
    bm_disp = bm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    bm_disp[bm_disp <= 0] = np.inf

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 1 * 5 * 5,
        P2=32 * 1 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    sgbm_disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    sgbm_disp[sgbm_disp <= 0] = np.inf
    return {"stereo_bm": bm_disp, "stereo_sgbm": sgbm_disp}


def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float) -> np.ndarray:
    depth = fx * baseline_m / disp
    depth[~np.isfinite(depth)] = np.inf
    depth[depth <= 0] = np.inf
    return depth


def run_model_batch(
    model: FoundationStereo,
    frames_left: List[np.ndarray],
    frames_right: List[np.ndarray],
    scale: float,
    hiera: int,
    valid_iters: int,
) -> List[np.ndarray]:
    if not frames_left:
        return []
    originals = [img.copy() for img in frames_left]
    resized_left = []
    resized_right = []
    shapes = []
    for left_img, right_img in zip(frames_left, frames_right):
        if scale != 1.0:
            left_img = cv2.resize(left_img, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
        resized_left.append(left_img)
        resized_right.append(right_img)
        shapes.append(left_img.shape[:2])

    same_shape = all(shape == shapes[0] for shape in shapes)
    if not same_shape:
        outputs = []
        for left_img, right_img, original in zip(frames_left, frames_right, originals):
            outputs.extend(run_model_batch(model, [left_img], [right_img], scale, hiera, valid_iters))
        return outputs

    left_tensor = torch.stack(
        [torch.as_tensor(img).float().permute(2, 0, 1) for img in resized_left], dim=0
    ).cuda()
    right_tensor = torch.stack(
        [torch.as_tensor(img).float().permute(2, 0, 1) for img in resized_right], dim=0
    ).cuda()
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    with torch.cuda.amp.autocast(True):
        if not hiera:
            disp_batch = model.forward(left_tensor, right_tensor, iters=valid_iters, test_mode=True)
        else:
            disp_batch = model.run_hierachical(
                left_tensor, right_tensor, iters=valid_iters, test_mode=True, small_ratio=0.5
            )

    disp_batch = padder.unpad(disp_batch.float()).data.cpu().numpy()
    outputs: List[np.ndarray] = []
    for disp, original in zip(disp_batch, originals):
        disp = disp.reshape(shapes[0][0], shapes[0][1])
        if scale != 1.0:
            disp = cv2.resize(
                disp,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) / scale
        outputs.append(disp)
    return outputs


def infer_baseline_from_depth(
    disp: np.ndarray,
    gt_depth_m: np.ndarray,
    fx: float,
    mask: np.ndarray,
) -> Optional[float]:
    valid = mask & np.isfinite(disp) & (disp > 1e-6)
    if valid.sum() < 500:
        return None
    baseline_samples = gt_depth_m[valid] * disp[valid] / fx
    baseline_samples = baseline_samples[np.isfinite(baseline_samples)]
    if baseline_samples.size == 0:
        return None
    q1, q3 = np.quantile(baseline_samples, [0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    robust = baseline_samples[(baseline_samples >= lo) & (baseline_samples <= hi)]
    if robust.size < 100:
        robust = baseline_samples
    return float(np.median(robust))


def save_comparison_vis(
    left_gray_rgb: np.ndarray,
    disp_vis: np.ndarray,
    pred_depth_m: np.ndarray,
    gt_depth_m: Optional[np.ndarray],
    out_path: str,
) -> None:
    pred_vis = vis_disparity(pred_depth_m, invalid_thres=np.inf)
    panels = [left_gray_rgb, disp_vis, pred_vis]
    if gt_depth_m is not None:
        gt_vis = vis_disparity(gt_depth_m, invalid_thres=np.inf)
        diff = np.abs(pred_depth_m - gt_depth_m)
        diff[~np.isfinite(diff)] = np.inf
        diff_vis = vis_disparity(diff, min_val=0.0, max_val=1.0, invalid_thres=np.inf)
        panels.extend([gt_vis, diff_vis])
    canvas = np.concatenate(panels, axis=1)
    imageio.imwrite(out_path, canvas)


def save_multi_method_vis(
    left_gray_rgb: np.ndarray,
    gt_depth_m: Optional[np.ndarray],
    method_depths: Dict[str, np.ndarray],
    out_path: str,
) -> None:
    title_map = {
        "foundation_stereo": "FoundationStereo",
        "stereo_bm": "StereoBM",
        "stereo_sgbm": "StereoSGBM",
        "raw_depth": "RawDepthRef",
    }
    panels = [left_gray_rgb]
    labels = ["Left IR"]
    if gt_depth_m is not None:
        panels.append(vis_disparity(gt_depth_m, invalid_thres=np.inf))
        labels.append("GT Depth")
    for key in ["foundation_stereo", "stereo_bm", "stereo_sgbm", "raw_depth"]:
        depth = method_depths.get(key)
        if depth is None:
            continue
        panels.append(vis_disparity(depth, invalid_thres=np.inf))
        labels.append(title_map.get(key, key))
        if gt_depth_m is not None and key != "raw_depth":
            diff = np.abs(depth - gt_depth_m)
            diff[~np.isfinite(diff)] = np.inf
            panels.append(vis_disparity(diff, min_val=0.0, max_val=1.0, invalid_thres=np.inf))
            labels.append(f"{title_map.get(key, key)} Err")

    annotated = []
    for img, label in zip(panels, labels):
        canvas = img.copy()
        cv2.putText(
            canvas,
            label,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        annotated.append(canvas)
    imageio.imwrite(out_path, np.concatenate(annotated, axis=1))


def save_summary_plots(all_metrics: List[Dict[str, float]], summary_dir: str) -> None:
    if plt is None:
        return
    method_labels = {
        "foundation_stereo": "FoundationStereo",
        "stereo_bm": "StereoBM",
        "stereo_sgbm": "StereoSGBM",
        "raw_depth": "RawDepthRef",
    }
    metric_keys = ["mae_m", "rmse_m", "abs_rel", "delta1"]
    method_to_metrics: Dict[str, Dict[str, List[float]]] = {}
    for item in all_metrics:
        method = item.get("method")
        if method is None:
            continue
        method_to_metrics.setdefault(method, {k: [] for k in metric_keys})
        for key in metric_keys:
            if key in item:
                method_to_metrics[method][key].append(item[key])

    if not method_to_metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    methods = [m for m in ["foundation_stereo", "stereo_sgbm", "stereo_bm", "raw_depth"] if m in method_to_metrics]
    for ax, metric_key in zip(axes, metric_keys):
        vals = []
        labels = []
        for method in methods:
            seq = method_to_metrics[method][metric_key]
            if seq:
                vals.append(float(np.mean(seq)))
                labels.append(method_labels.get(method, method))
        if not vals:
            ax.axis("off")
            continue
        bars = ax.bar(labels, vals, color=["#2E86DE", "#27AE60", "#F39C12", "#7F8C8D"][: len(vals)])
        ax.set_title(metric_key)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    fig.suptitle("Depth Estimation Error Summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(summary_dir, "error_summary.png"), dpi=180)
    plt.close(fig)

    frame_ids = [item["frame_id"] for item in all_metrics if item.get("method") == "foundation_stereo"]
    if frame_ids:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        for ax, metric_key in zip(axes, ["mae_m", "rmse_m", "delta1"]):
            for method in methods:
                seq = [item.get(metric_key) for item in all_metrics if item.get("method") == method and metric_key in item]
                if len(seq) == len(frame_ids):
                    ax.plot(frame_ids, seq, marker="o", label=method_labels.get(method, method))
            ax.set_ylabel(metric_key)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
        axes[-1].set_xlabel("Frame ID")
        fig.suptitle("Per-frame Accuracy Curves", fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(summary_dir, "per_frame_metrics.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260403_202138",
        type=str,
    )
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--out_dir",
        default=f"{code_dir}/../output/orbbec_demo",
        type=str,
        help="directory to save results",
    )
    parser.add_argument("--scale", default=1.0, type=float, help="downscale factor, must be <= 1")
    parser.add_argument("--hiera", default=1, type=int, help="hierarchical inference for high-resolution images")
    parser.add_argument("--valid_iters", type=int, default=32, help="number of refinement iterations")
    parser.add_argument("--batch_size", type=int, default=1, help="number of stereo pairs inferred together")
    parser.add_argument(
        "--device_name",
        type=str,
        default="gemini_435le",
        help="camera preset name; used to set safer defaults for baseline and depth scale",
    )
    parser.add_argument(
        "--baseline_m",
        type=float,
        default=-1.0,
        help="stereo baseline in meters; if > 0, overrides camera preset and depth-based estimation",
    )
    parser.add_argument(
        "--fallback_baseline_m",
        type=float,
        default=-1.0,
        help="manual fallback baseline in meters; if <= 0, use camera preset baseline",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=-1.0,
        help="multiply raw depth values by this factor to get meters; if <= 0, use camera preset",
    )
    parser.add_argument("--max_frames", type=int, default=-1, help="limit number of frames")
    parser.add_argument("--start_index", type=int, default=0, help="start frame index after sorting")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="run prediction every n frames after sorting; 1 means use every frame",
    )
    parser.add_argument("--z_far", type=float, default=10.0, help="max depth retained in point cloud")
    parser.add_argument("--remove_invisible", type=int, default=1, help="mask points without right-image support")
    parser.add_argument("--denoise_cloud", type=int, default=1, help="apply point cloud denoising")
    parser.add_argument("--denoise_nb_points", type=int, default=30)
    parser.add_argument("--denoise_radius", type=float, default=0.03)
    parser.add_argument("--save_npz", type=int, default=1, help="save disparity/depth arrays")
    parser.add_argument("--enable_classic_methods", type=int, default=1, help="run StereoBM/SGBM comparison")
    parser.add_argument("--enable_pointcloud", type=int, default=1, help="export predicted point clouds")
    parser.add_argument("--enable_plots", type=int, default=1, help="generate summary plots")
    args = parser.parse_args()

    assert args.scale <= 1.0, "scale must be <= 1"
    assert args.frame_stride >= 1, "frame_stride must be >= 1"
    assert args.batch_size >= 1, "batch_size must be >= 1"

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "vis")
    analysis_dir = os.path.join(args.out_dir, "analysis")
    depth_dir = os.path.join(args.out_dir, "depth")
    cloud_dir = os.path.join(args.out_dir, "cloud")
    metric_dir = os.path.join(args.out_dir, "metrics")
    npz_dir = os.path.join(args.out_dir, "npz")
    for folder in [vis_dir, analysis_dir, depth_dir, cloud_dir, metric_dir, npz_dir]:
        os.makedirs(folder, exist_ok=True)

    meta_path = os.path.join(args.data_dir, "camera_intrinsics.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    preset = get_device_preset(args.device_name)
    if preset is None:
        logging.warning("Unknown device preset '%s'. Falling back to script defaults.", args.device_name)
        preset = {
            "display_name": args.device_name,
            "stereo_baseline_m": 0.095,
            "depth_scale_m": 0.001,
            "notes": "Using generic defaults.",
        }

    depth_scale_m = args.depth_scale if args.depth_scale > 0 else preset["depth_scale_m"]
    preset_baseline_m = preset["stereo_baseline_m"]
    fallback_baseline_m = args.fallback_baseline_m if args.fallback_baseline_m > 0 else preset_baseline_m

    ir_intrinsic = meta["streams"]["IR Left"]["intrinsic"]
    depth_intrinsic = meta["streams"]["Depth"]["intrinsic"]
    color_intrinsic = meta["streams"]["Color"]["intrinsic"]
    k_ir = build_k(ir_intrinsic)
    k_depth = build_k(depth_intrinsic)
    k_color = build_k(color_intrinsic)

    transform_meta = meta["pipeline_camera_param"]["transform"]
    t_depth_to_color = make_transform_matrix(
        transform_meta["rot"], transform_meta["transform"], translation_scale=0.001
    )

    left_files = list_pngs(os.path.join(args.data_dir, "ir_left"))
    right_files = list_pngs(os.path.join(args.data_dir, "ir_right"))
    color_files = list_pngs(os.path.join(args.data_dir, "color"))
    depth_npy_paths = {}
    depth_dir_input = os.path.join(args.data_dir, "depth")
    if os.path.isdir(depth_dir_input):
        for name in sorted(os.listdir(depth_dir_input)):
            if name.lower().endswith(".npy"):
                depth_npy_paths[os.path.splitext(name)[0]] = os.path.join(depth_dir_input, name)

    common_stems = sorted(set(left_files) & set(right_files))
    if not common_stems:
        raise RuntimeError("No matching frame ids found in ir_left and ir_right")

    common_stems = common_stems[args.start_index :]
    common_stems = common_stems[:: args.frame_stride]
    if args.max_frames > 0:
        common_stems = common_stems[: args.max_frames]

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for key, value in vars(args).items():
        cfg[key] = value
    cfg = OmegaConf.create(cfg)

    logging.info("Loading model from %s", ckpt_dir)
    logging.info(
        "Using device preset: %s | preset_baseline=%.6fm | depth_scale=%.6f m/unit",
        preset["display_name"],
        preset_baseline_m,
        depth_scale_m,
    )
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.cuda()
    model.eval()

    all_metrics = []
    baseline_history = []
    running_baseline = args.baseline_m if args.baseline_m > 0 else None

    logging.info(
        "Processing %d frame pairs from %s | batch_size=%d",
        len(common_stems),
        args.data_dir,
        args.batch_size,
    )

    processed_count = 0
    for chunk_start in range(0, len(common_stems), args.batch_size):
        chunk_stems = common_stems[chunk_start : chunk_start + args.batch_size]
        chunk_left = [load_gray_as_rgb(left_files[stem]) for stem in chunk_stems]
        chunk_right = [load_gray_as_rgb(right_files[stem]) for stem in chunk_stems]
        chunk_disp = run_model_batch(
            model,
            chunk_left,
            chunk_right,
            scale=args.scale,
            hiera=args.hiera,
            valid_iters=args.valid_iters,
        )

        for stem, left_img_loaded, right_img_loaded, disp in zip(chunk_stems, chunk_left, chunk_right, chunk_disp):
            processed_count += 1
            idx = processed_count
            left_path = left_files[stem]
            right_path = right_files[stem]
            color_path = color_files.get(stem)
            depth_path = depth_npy_paths.get(stem)

            left_gray_vis = left_img_loaded.copy()
            left_gray = cv2.cvtColor(left_gray_vis, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img_loaded, cv2.COLOR_RGB2GRAY)

            if args.remove_invisible:
                yy, xx = np.meshgrid(
                    np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
                )
                invalid = (xx - disp) < 0
                disp = disp.copy()
                disp[invalid] = np.inf

            gt_depth_m = None
            valid_gt_mask = None
            if depth_path and os.path.exists(depth_path):
                gt_depth_m = depth_to_meters(np.load(depth_path), depth_scale_m)
                if gt_depth_m.shape[:2] != disp.shape[:2]:
                    gt_depth_m = cv2.resize(
                        gt_depth_m, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                valid_gt_mask = np.isfinite(gt_depth_m) & (gt_depth_m > 0.05) & (gt_depth_m < args.z_far)
            else:
                valid_gt_mask = np.zeros_like(disp, dtype=bool)

            fx = k_ir[0, 0]
            if args.scale != 1.0:
                fx *= args.scale
            baseline_source = "manual_input" if args.baseline_m > 0 else "device_preset"
            if running_baseline is None and gt_depth_m is not None and args.baseline_m <= 0:
                estimated = infer_baseline_from_depth(disp, gt_depth_m, fx, valid_gt_mask)
                if estimated is not None:
                    baseline_history.append(estimated)
                    running_baseline = float(np.median(np.array(baseline_history, dtype=np.float32)))
                    baseline_source = "estimated_from_depth"

            if args.baseline_m > 0:
                baseline_m = args.baseline_m
            elif running_baseline is not None:
                baseline_m = running_baseline
                baseline_source = "estimated_from_depth"
            else:
                baseline_m = fallback_baseline_m
                baseline_source = "device_preset" if args.fallback_baseline_m <= 0 else "manual_fallback"

            pred_depth_m = fx * baseline_m / disp
            pred_depth_m[~np.isfinite(pred_depth_m)] = np.inf
            pred_depth_m[pred_depth_m <= 0] = np.inf

            method_depths = {"foundation_stereo": pred_depth_m}
            if args.enable_classic_methods:
                classic_disps = compute_classic_disparities(left_gray, right_gray)
                method_depths["stereo_bm"] = disparity_to_depth(classic_disps["stereo_bm"], fx, baseline_m)
                method_depths["stereo_sgbm"] = disparity_to_depth(classic_disps["stereo_sgbm"], fx, baseline_m)
            if gt_depth_m is not None:
                method_depths["raw_depth"] = gt_depth_m

            frame_metrics = []
            foundation_metrics = {
                "frame_id": stem,
                "method": "foundation_stereo",
                "baseline_used_m": float(baseline_m),
                "baseline_source": baseline_source,
                "device_name": preset["display_name"],
                "depth_scale_m_per_unit": float(depth_scale_m),
            }
            if gt_depth_m is not None:
                foundation_metrics["gt_valid_ratio"] = float(valid_gt_mask.mean())
            if gt_depth_m is not None:
                valid_compare_mask = (
                    np.isfinite(method_depths["foundation_stereo"])
                    & np.isfinite(gt_depth_m)
                    & (method_depths["foundation_stereo"] > 0.05)
                    & (method_depths["foundation_stereo"] < args.z_far)
                    & valid_gt_mask
                )
                if valid_compare_mask.sum() > 500:
                    foundation_metrics.update(compute_metrics(method_depths["foundation_stereo"], gt_depth_m, valid_compare_mask))
            foundation_metrics.update(compute_depth_stats(method_depths["foundation_stereo"], args.z_far))
            frame_metrics.append(foundation_metrics)

            if gt_depth_m is not None and args.enable_classic_methods:
                for method_name in ["stereo_bm", "stereo_sgbm"]:
                    depth_pred = method_depths[method_name]
                    method_mask = (
                        np.isfinite(depth_pred)
                        & np.isfinite(gt_depth_m)
                        & (depth_pred > 0.05)
                        & (depth_pred < args.z_far)
                        & valid_gt_mask
                    )
                    metric_item = {
                        "frame_id": stem,
                        "method": method_name,
                        "baseline_used_m": float(baseline_m),
                        "device_name": preset["display_name"],
                    }
                    if method_mask.sum() > 500:
                        metric_item.update(compute_metrics(depth_pred, gt_depth_m, method_mask))
                    metric_item.update(compute_depth_stats(depth_pred, args.z_far))
                    frame_metrics.append(metric_item)

                raw_metric = {
                    "frame_id": stem,
                    "method": "raw_depth",
                    "device_name": preset["display_name"],
                    "reference_role": "ground_truth_reference",
                }
                raw_metric.update(compute_depth_stats(gt_depth_m, args.z_far))
                raw_metric.update(
                    {
                        "mae_m": 0.0,
                        "rmse_m": 0.0,
                        "abs_rel": 0.0,
                        "delta1": 1.0,
                        "num_valid": int(valid_gt_mask.sum()),
                        "pred_mean_m": float(gt_depth_m[valid_gt_mask].mean()) if valid_gt_mask.any() else 0.0,
                        "gt_mean_m": float(gt_depth_m[valid_gt_mask].mean()) if valid_gt_mask.any() else 0.0,
                    }
                )
                frame_metrics.append(raw_metric)

            all_metrics.extend(frame_metrics)

            disp_vis = vis_disparity(disp, invalid_thres=np.inf)
            save_comparison_vis(
                left_gray_vis,
                disp_vis,
                pred_depth_m,
                gt_depth_m,
                os.path.join(vis_dir, f"{stem}.png"),
            )
            if args.enable_classic_methods:
                save_multi_method_vis(
                    left_gray_vis,
                    gt_depth_m,
                    method_depths,
                    os.path.join(analysis_dir, f"{stem}_methods.png"),
                )

            np.save(os.path.join(depth_dir, f"{stem}_pred_depth_m.npy"), pred_depth_m.astype(np.float32))
            if args.save_npz:
                payload = {
                    "disp": disp.astype(np.float32),
                    "pred_depth_m": pred_depth_m.astype(np.float32),
                }
                if args.enable_classic_methods:
                    payload["bm_depth_m"] = method_depths["stereo_bm"].astype(np.float32)
                    payload["sgbm_depth_m"] = method_depths["stereo_sgbm"].astype(np.float32)
                if gt_depth_m is not None:
                    payload["gt_depth_m"] = gt_depth_m.astype(np.float32)
                np.savez_compressed(os.path.join(npz_dir, f"{stem}.npz"), **payload)

            if args.enable_pointcloud:
                color_img = load_color_rgb(color_path) if color_path and os.path.exists(color_path) else left_gray_vis
                if color_img.shape[:2] != left_gray_vis.shape[:2] and color_path:
                    color_img = cv2.resize(color_img, (left_gray_vis.shape[1], left_gray_vis.shape[0]), interpolation=cv2.INTER_LINEAR)

                k_depth_use = k_depth.copy()
                if pred_depth_m.shape[:2] != (depth_intrinsic["height"], depth_intrinsic["width"]):
                    scale_x = pred_depth_m.shape[1] / depth_intrinsic["width"]
                    scale_y = pred_depth_m.shape[0] / depth_intrinsic["height"]
                    k_depth_use[0, 0] *= scale_x
                    k_depth_use[1, 1] *= scale_y
                    k_depth_use[0, 2] *= scale_x
                    k_depth_use[1, 2] *= scale_y

                xyz_map = depth2xyzmap(pred_depth_m, k_depth_use)
                if color_path and os.path.exists(color_path):
                    colors = project_points_to_color(xyz_map, color_img, k_color, t_depth_to_color)
                else:
                    colors = left_gray_vis.reshape(-1, 3)
                pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), colors.reshape(-1, 3))

                points = np.asarray(pcd.points)
                keep_mask = np.isfinite(points[:, 2]) & (points[:, 2] > 0) & (points[:, 2] <= args.z_far)
                keep_ids = np.where(keep_mask)[0]
                pcd = pcd.select_by_index(keep_ids.tolist())

                o3d.io.write_point_cloud(os.path.join(cloud_dir, f"{stem}_pred_cloud.ply"), pcd)

                if args.denoise_cloud and len(np.asarray(pcd.points)) > 0:
                    _, inlier_idx = pcd.remove_radius_outlier(
                        nb_points=args.denoise_nb_points, radius=args.denoise_radius
                    )
                    pcd_denoise = pcd.select_by_index(inlier_idx)
                    o3d.io.write_point_cloud(
                        os.path.join(cloud_dir, f"{stem}_pred_cloud_denoise.ply"), pcd_denoise
                    )

            metric_path = os.path.join(metric_dir, f"{stem}.json")
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump({"frame_id": stem, "methods": frame_metrics}, f, indent=2, ensure_ascii=False)

            logging.info(
                "[%d/%d] frame=%s baseline=%.6fm FS_valid=%s FS_mae=%s FS_rmse=%s",
                idx,
                len(common_stems),
                stem,
                baseline_m,
                foundation_metrics.get("num_valid", 0),
                f"{foundation_metrics['mae_m']:.4f}" if "mae_m" in foundation_metrics else "n/a",
                f"{foundation_metrics['rmse_m']:.4f}" if "rmse_m" in foundation_metrics else "n/a",
            )

    summary = {
        "num_frames": len(common_stems),
        "device_name": preset["display_name"],
        "device_notes": preset["notes"],
        "baseline_input_m": args.baseline_m,
        "preset_baseline_m": preset_baseline_m,
        "fallback_baseline_m": fallback_baseline_m,
        "depth_scale_m_per_unit": depth_scale_m,
        "estimated_baseline_m": float(np.median(np.array(baseline_history))) if baseline_history else None,
    }

    method_summary = {}
    methods = sorted({m["method"] for m in all_metrics if "method" in m})
    valid_metric_frames = [m for m in all_metrics if "mae_m" in m]
    if valid_metric_frames:
        for method in methods:
            frames = [m for m in all_metrics if m.get("method") == method and "mae_m" in m]
            if not frames:
                continue
            method_summary[method] = {"num_frames": len(frames)}
            for key in ["mae_m", "rmse_m", "abs_rel", "delta1", "pred_mean_m", "gt_mean_m", "valid_ratio"]:
                vals = [m[key] for m in frames if key in m]
                if vals:
                    method_summary[method][f"mean_{key}"] = float(np.mean(vals))
        summary["num_frames_with_depth_compare"] = len(
            {m["frame_id"] for m in valid_metric_frames if m.get("method") == "foundation_stereo"}
        )
    else:
        summary["num_frames_with_depth_compare"] = 0
    summary["method_summary"] = method_summary

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "analysis_report.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_frame_metrics": all_metrics}, f, indent=2, ensure_ascii=False)

    if args.enable_plots:
        save_summary_plots(all_metrics, args.out_dir)

    logging.info("Finished. Results saved to %s", args.out_dir)
    logging.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
