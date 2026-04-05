import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d


def set_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_k(intrinsic: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [intrinsic["fx"], 0.0, intrinsic["cx"]],
            [0.0, intrinsic["fy"], intrinsic["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_transform_matrix(rot_flat: List[float], trans_mm: List[float], translation_scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(rot_flat, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.array(trans_mm, dtype=np.float32) * translation_scale
    return transform


def list_indexed_files(folder: str, suffix: str = ".png") -> Dict[int, str]:
    files: Dict[int, str] = {}
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(suffix):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            files[int(stem)] = os.path.join(folder, name)
    return files


def parse_optional_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass
class FrameRow:
    frame_index: int
    system_time: Optional[float]
    color_timestamp: Optional[int]
    depth_timestamp: Optional[int]
    ir_timestamp: Optional[int]
    ir_left_timestamp: Optional[int]
    ir_right_timestamp: Optional[int]

    @property
    def pose_timestamp(self) -> Optional[int]:
        for ts in [self.ir_left_timestamp, self.ir_right_timestamp, self.color_timestamp, self.depth_timestamp]:
            if ts is not None:
                return ts
        return None


@dataclass
class ImuSample:
    timestamp: int
    sensor_type: str
    acc: np.ndarray
    gyro: Optional[np.ndarray]


def load_frame_rows(csv_path: str) -> List[FrameRow]:
    rows: List[FrameRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                FrameRow(
                    frame_index=int(row["frame_index"]),
                    system_time=parse_float(row.get("system_time", "")),
                    color_timestamp=parse_optional_int(row.get("color_timestamp", "")),
                    depth_timestamp=parse_optional_int(row.get("depth_timestamp", "")),
                    ir_timestamp=parse_optional_int(row.get("ir_timestamp", "")),
                    ir_left_timestamp=parse_optional_int(row.get("ir_left_timestamp", "")),
                    ir_right_timestamp=parse_optional_int(row.get("ir_right_timestamp", "")),
                )
            )
    return rows


def load_imu_samples(csv_path: str) -> List[ImuSample]:
    samples: List[ImuSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            acc = np.array(
                [
                    parse_float(row.get("acc_x", "")) or 0.0,
                    parse_float(row.get("acc_y", "")) or 0.0,
                    parse_float(row.get("acc_z", "")) or 0.0,
                ],
                dtype=np.float64,
            )
            gyro_vals = [parse_float(row.get("gyro_x", "")), parse_float(row.get("gyro_y", "")), parse_float(row.get("gyro_z", ""))]
            gyro = None if any(v is None for v in gyro_vals) else np.array(gyro_vals, dtype=np.float64)
            samples.append(
                ImuSample(
                    timestamp=int(float(row["timestamp"])),
                    sensor_type=row["sensor_type"],
                    acc=acc,
                    gyro=gyro,
                )
            )
    return samples


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def exp_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(omega)
    axis = omega / theta
    k = skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return exp_so3(v * np.pi)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3, dtype=np.float64)
    vx = skew(v)
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def project_points_to_color(
    xyz_depth: np.ndarray,
    color_img: np.ndarray,
    k_color: np.ndarray,
    t_depth_to_color: np.ndarray,
) -> np.ndarray:
    points = xyz_depth.reshape(-1, 3)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    if not np.any(valid):
        return colors
    points_h = np.concatenate([points[valid], np.ones((valid.sum(), 1), dtype=np.float32)], axis=1)
    points_color = (t_depth_to_color @ points_h.T).T[:, :3]
    z = points_color[:, 2]
    valid_z = z > 1e-6
    if not np.any(valid_z):
        return colors
    pc = points_color[valid_z]
    u = np.round(pc[:, 0] * k_color[0, 0] / pc[:, 2] + k_color[0, 2]).astype(np.int32)
    v = np.round(pc[:, 1] * k_color[1, 1] / pc[:, 2] + k_color[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < color_img.shape[1]) & (v >= 0) & (v < color_img.shape[0])
    valid_ids = np.where(valid)[0]
    valid_z_ids = valid_ids[valid_z]
    colors[valid_z_ids[inside]] = color_img[v[inside], u[inside], :3]
    return colors


def depth_to_xyz(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth
    x = (xs - k[0, 2]) * z / k[0, 0]
    y = (ys - k[1, 2]) * z / k[1, 1]
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    invalid = ~np.isfinite(z) | (z <= 0)
    xyz[invalid] = 0
    return xyz


def choose_color_frame(
    current_row: FrameRow,
    frame_rows: Dict[int, FrameRow],
    color_files: Dict[int, str],
) -> Optional[int]:
    if current_row.frame_index in color_files:
        return current_row.frame_index
    if current_row.color_timestamp is not None:
        candidates = []
        for idx, row in frame_rows.items():
            if idx not in color_files or row.color_timestamp is None:
                continue
            candidates.append((abs(row.color_timestamp - current_row.color_timestamp), idx))
        if candidates:
            candidates.sort()
            return candidates[0][1]
    return None


def integrate_imu_poses(
    imu_samples: List[ImuSample],
    pose_timestamps: List[int],
    gravity_mps2: float = 9.81,
    accel_deadband: float = 0.15,
) -> Dict[int, np.ndarray]:
    valid_samples = [s for s in imu_samples if s.gyro is not None]
    if not valid_samples:
        raise RuntimeError("No IMU accel+gyro samples available.")

    pose_timestamps = sorted(set(pose_timestamps))
    gravity_body = np.mean([s.acc for s in valid_samples[: min(50, len(valid_samples))]], axis=0)
    gravity_world = np.array([0.0, 0.0, -gravity_mps2], dtype=np.float64)
    r_world_body = rotation_from_a_to_b(gravity_body, gravity_world)
    pos = np.zeros(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)

    poses: Dict[int, np.ndarray] = {}
    target_idx = 0
    last_ts = valid_samples[0].timestamp

    for sample in valid_samples:
        ts = sample.timestamp
        dt = max((ts - last_ts) * 1e-3, 0.0)
        last_ts = ts
        if dt > 0:
            r_world_body = r_world_body @ exp_so3(sample.gyro * dt)
            acc_world = r_world_body @ sample.acc
            lin_acc = acc_world - gravity_world
            lin_norm = np.linalg.norm(lin_acc)
            if lin_norm < accel_deadband:
                lin_acc[:] = 0.0
            pos = pos + vel * dt + 0.5 * lin_acc * dt * dt
            vel = vel + lin_acc * dt

        while target_idx < len(pose_timestamps) and ts >= pose_timestamps[target_idx]:
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = r_world_body.astype(np.float32)
            transform[:3, 3] = pos.astype(np.float32)
            poses[pose_timestamps[target_idx]] = transform
            target_idx += 1

    last_pose = np.eye(4, dtype=np.float32)
    last_pose[:3, :3] = r_world_body.astype(np.float32)
    last_pose[:3, 3] = pos.astype(np.float32)
    while target_idx < len(pose_timestamps):
        poses[pose_timestamps[target_idx]] = last_pose.copy()
        target_idx += 1
    return poses


def save_trajectory_plot(points: np.ndarray, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if len(points) == 0:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", linewidth=1.5)
    ax.set_title("IMU Integrated Camera Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100",
        type=str,
    )
    parser.add_argument(
        "--depth_dir",
        default="output/depth",
        type=str,
        help="relative or absolute directory that stores *_pred_depth_m.npy",
    )
    parser.add_argument(
        "--out_dir",
        default="output/reconstruction",
        type=str,
        help="relative or absolute output directory",
    )
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--z_near", type=float, default=0.1)
    parser.add_argument("--z_far", type=float, default=8.0)
    parser.add_argument("--depth_sample_stride", type=int, default=4, help="pixel stride when lifting depth to 3D")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="voxel downsample size for merged cloud")
    parser.add_argument("--imu_accel_deadband", type=float, default=0.15)
    parser.add_argument("--save_per_frame_cloud", type=int, default=0)
    args = parser.parse_args()

    assert args.frame_stride >= 1
    assert args.depth_sample_stride >= 1

    set_logging()

    data_dir = args.data_dir
    depth_dir = args.depth_dir if os.path.isabs(args.depth_dir) else os.path.join(data_dir, args.depth_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(data_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    per_frame_dir = os.path.join(out_dir, "per_frame_clouds")
    if args.save_per_frame_cloud:
        os.makedirs(per_frame_dir, exist_ok=True)

    with open(os.path.join(data_dir, "camera_intrinsics.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    k_depth = build_k(meta["pipeline_camera_param"]["depth_intrinsic"])
    k_color = build_k(meta["pipeline_camera_param"]["rgb_intrinsic"])
    t_depth_to_color = make_transform_matrix(
        meta["pipeline_camera_param"]["transform"]["rot"],
        meta["pipeline_camera_param"]["transform"]["transform"],
        translation_scale=0.001,
    )

    frame_rows_list = load_frame_rows(os.path.join(data_dir, "frame_timestamps.csv"))
    frame_rows = {row.frame_index: row for row in frame_rows_list}
    imu_samples = load_imu_samples(os.path.join(data_dir, "imu_data.csv"))

    color_files = list_indexed_files(os.path.join(data_dir, "color"), suffix=".png")
    pred_depth_files = {}
    if os.path.isdir(depth_dir):
        for name in sorted(os.listdir(depth_dir)):
            if not name.endswith("_pred_depth_m.npy"):
                continue
            stem = name.replace("_pred_depth_m.npy", "")
            if stem.isdigit():
                pred_depth_files[int(stem)] = os.path.join(depth_dir, name)

    frame_indices = sorted(pred_depth_files.keys())
    frame_indices = frame_indices[:: args.frame_stride]
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise RuntimeError(f"No predicted depth files found in {depth_dir}")

    selected_rows: List[Tuple[int, FrameRow, int]] = []
    pose_timestamps: List[int] = []
    for frame_index in frame_indices:
        row = frame_rows.get(frame_index)
        if row is None or row.pose_timestamp is None:
            logging.warning("Skip frame %06d: missing frame timestamp row or pose timestamp", frame_index)
            continue
        color_index = choose_color_frame(row, frame_rows, color_files)
        if color_index is None:
            logging.warning("Skip frame %06d: missing usable color frame", frame_index)
            continue
        selected_rows.append((frame_index, row, color_index))
        pose_timestamps.append(row.pose_timestamp)

    if not selected_rows:
        raise RuntimeError("No frame could be aligned across predicted depth, timestamps, and color.")

    pose_map = integrate_imu_poses(
        imu_samples,
        pose_timestamps,
        accel_deadband=args.imu_accel_deadband,
    )

    all_points = []
    all_colors = []
    trajectory = []
    kept_frames = []

    for frame_index, row, color_index in selected_rows:
        pose = pose_map.get(row.pose_timestamp)
        if pose is None:
            logging.warning("Skip frame %06d: IMU pose not available", frame_index)
            continue

        depth = np.load(pred_depth_files[frame_index]).astype(np.float32)
        if depth.ndim != 2:
            logging.warning("Skip frame %06d: invalid depth shape %s", frame_index, depth.shape)
            continue

        color = imageio.imread(color_files[color_index])
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=2)
        color = color[..., :3]

        if depth.shape[:2] != (meta["pipeline_camera_param"]["depth_intrinsic"]["height"], meta["pipeline_camera_param"]["depth_intrinsic"]["width"]):
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            scale_x = depth.shape[1] / meta["pipeline_camera_param"]["depth_intrinsic"]["width"]
            scale_y = depth.shape[0] / meta["pipeline_camera_param"]["depth_intrinsic"]["height"]
            k_depth_use = k_depth.copy()
            k_depth_use[0, 0] *= scale_x
            k_depth_use[1, 1] *= scale_y
            k_depth_use[0, 2] *= scale_x
            k_depth_use[1, 2] *= scale_y
        else:
            k_depth_use = k_depth

        valid = np.isfinite(depth) & (depth > args.z_near) & (depth < args.z_far)
        if not np.any(valid):
            logging.warning("Skip frame %06d: no valid predicted depth", frame_index)
            continue

        xyz_depth = depth_to_xyz(depth, k_depth_use)
        colors = project_points_to_color(xyz_depth, color, k_color, t_depth_to_color)

        xyz = xyz_depth[:: args.depth_sample_stride, :: args.depth_sample_stride].reshape(-1, 3)
        rgb = colors.reshape(depth.shape[0], depth.shape[1], 3)[:: args.depth_sample_stride, :: args.depth_sample_stride].reshape(-1, 3)
        valid_pts = np.isfinite(xyz[:, 2]) & (xyz[:, 2] > args.z_near) & (xyz[:, 2] < args.z_far)
        xyz = xyz[valid_pts]
        rgb = rgb[valid_pts]
        if len(xyz) == 0:
            continue

        xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
        xyz_world = (pose @ xyz_h.T).T[:, :3]

        all_points.append(xyz_world.astype(np.float32))
        all_colors.append(rgb.astype(np.uint8))
        trajectory.append(pose[:3, 3].copy())
        kept_frames.append(
            {
                "frame_index": frame_index,
                "color_index": color_index,
                "pose_timestamp": row.pose_timestamp,
                "num_points": int(len(xyz_world)),
                "translation_m": pose[:3, 3].astype(float).tolist(),
            }
        )

        if args.save_per_frame_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_world.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float64) / 255.0))
            o3d.io.write_point_cloud(os.path.join(per_frame_dir, f"{frame_index:06d}.ply"), pcd)

        logging.info(
            "Frame %06d | color %06d | pose_ts=%s | points=%d",
            frame_index,
            color_index,
            row.pose_timestamp,
            len(xyz_world),
        )

    if not all_points:
        raise RuntimeError("No valid point cloud points were reconstructed.")

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points.astype(np.float64))
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors.astype(np.float64) / 255.0)
    if args.voxel_size > 0:
        merged_pcd = merged_pcd.voxel_down_sample(args.voxel_size)

    merged_path = os.path.join(out_dir, "reconstruction_global.ply")
    o3d.io.write_point_cloud(merged_path, merged_pcd)

    traj_points = np.array(trajectory, dtype=np.float32) if trajectory else np.zeros((0, 3), dtype=np.float32)
    np.save(os.path.join(out_dir, "trajectory.npy"), traj_points)
    save_trajectory_plot(traj_points, os.path.join(out_dir, "trajectory.png"))

    summary = {
        "data_dir": data_dir,
        "depth_dir": depth_dir,
        "num_depth_frames_requested": len(frame_indices),
        "num_depth_frames_used": len(kept_frames),
        "num_points_before_merge": int(len(merged_points)),
        "num_points_after_voxel": int(len(np.asarray(merged_pcd.points))),
        "voxel_size": args.voxel_size,
        "depth_sample_stride": args.depth_sample_stride,
        "z_near": args.z_near,
        "z_far": args.z_far,
        "imu_pose_assumption": "IMU pose is used as depth camera pose with identity IMU-to-depth extrinsic.",
        "depth_to_color_translation_m": t_depth_to_color[:3, 3].astype(float).tolist(),
        "frames": kept_frames,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Saved merged reconstruction to %s", merged_path)
    logging.info("Used %d frames for reconstruction", len(kept_frames))


if __name__ == "__main__":
    main()
