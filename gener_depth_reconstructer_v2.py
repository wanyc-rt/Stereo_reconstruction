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
        dtype=np.float64,
    )


def make_transform_matrix(rot_flat: List[float], trans_mm: List[float], translation_scale: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(rot_flat, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.array(trans_mm, dtype=np.float64) * translation_scale
    return transform


def list_indexed_files(folder: str, suffix: str) -> Dict[int, str]:
    files: Dict[int, str] = {}
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(suffix):
            continue
        stem = os.path.splitext(name)[0]
        if suffix == ".npy" and stem.endswith("_pred_depth_m"):
            stem = stem.replace("_pred_depth_m", "")
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
    acc: np.ndarray
    gyro: Optional[np.ndarray]


@dataclass
class ReconFrame:
    frame_index: int
    color_index: int
    pose_timestamp: int
    color: np.ndarray
    depth_color_m: np.ndarray


def load_frame_rows(csv_path: str) -> Dict[int, FrameRow]:
    rows: Dict[int, FrameRow] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = FrameRow(
                frame_index=int(row["frame_index"]),
                system_time=parse_float(row.get("system_time", "")),
                color_timestamp=parse_optional_int(row.get("color_timestamp", "")),
                depth_timestamp=parse_optional_int(row.get("depth_timestamp", "")),
                ir_left_timestamp=parse_optional_int(row.get("ir_left_timestamp", "")),
                ir_right_timestamp=parse_optional_int(row.get("ir_right_timestamp", "")),
            )
            rows[fr.frame_index] = fr
    return rows


def load_imu_samples(csv_path: str) -> List[ImuSample]:
    samples: List[ImuSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gyro_vals = [parse_float(row.get("gyro_x", "")), parse_float(row.get("gyro_y", "")), parse_float(row.get("gyro_z", ""))]
            gyro = None if any(v is None for v in gyro_vals) else np.array(gyro_vals, dtype=np.float64)
            samples.append(
                ImuSample(
                    timestamp=int(float(row["timestamp"])),
                    acc=np.array(
                        [
                            parse_float(row.get("acc_x", "")) or 0.0,
                            parse_float(row.get("acc_y", "")) or 0.0,
                            parse_float(row.get("acc_z", "")) or 0.0,
                        ],
                        dtype=np.float64,
                    ),
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


def integrate_imu_poses(imu_samples: List[ImuSample], pose_timestamps: List[int], gravity_mps2: float = 9.81) -> Dict[int, np.ndarray]:
    valid = [s for s in imu_samples if s.gyro is not None]
    if not valid:
        return {}
    pose_timestamps = sorted(set(pose_timestamps))
    gravity_body = np.mean([s.acc for s in valid[: min(50, len(valid))]], axis=0)
    gravity_world = np.array([0.0, 0.0, -gravity_mps2], dtype=np.float64)
    r_world_body = rotation_from_a_to_b(gravity_body, gravity_world)
    pos = np.zeros(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    poses: Dict[int, np.ndarray] = {}
    target_idx = 0
    last_ts = valid[0].timestamp
    for sample in valid:
        dt = max((sample.timestamp - last_ts) * 1e-3, 0.0)
        last_ts = sample.timestamp
        if dt > 0:
            r_world_body = r_world_body @ exp_so3(sample.gyro * dt)
            acc_world = r_world_body @ sample.acc
            lin_acc = acc_world - gravity_world
            pos = pos + vel * dt + 0.5 * lin_acc * dt * dt
            vel = vel + lin_acc * dt
        while target_idx < len(pose_timestamps) and sample.timestamp >= pose_timestamps[target_idx]:
            t = np.eye(4, dtype=np.float64)
            t[:3, :3] = r_world_body
            t[:3, 3] = pos
            poses[pose_timestamps[target_idx]] = t
            target_idx += 1
    last_pose = np.eye(4, dtype=np.float64)
    last_pose[:3, :3] = r_world_body
    last_pose[:3, 3] = pos
    while target_idx < len(pose_timestamps):
        poses[pose_timestamps[target_idx]] = last_pose.copy()
        target_idx += 1
    return poses


def choose_color_frame(row: FrameRow, frame_rows: Dict[int, FrameRow], color_files: Dict[int, str]) -> Optional[int]:
    if row.frame_index in color_files:
        return row.frame_index
    if row.color_timestamp is None:
        target_ts = row.pose_timestamp
    else:
        target_ts = row.color_timestamp
    if target_ts is None:
        return None

    candidates = []
    for idx, other in frame_rows.items():
        if idx not in color_files:
            continue
        ref_ts = other.color_timestamp
        if ref_ts is None:
            ref_ts = other.pose_timestamp
        if ref_ts is None:
            continue
        candidates.append((abs(ref_ts - target_ts), idx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def align_depth_to_color(
    depth_m: np.ndarray,
    k_depth: np.ndarray,
    k_color: np.ndarray,
    t_depth_to_color: np.ndarray,
    color_shape: Tuple[int, int],
    z_near: float,
    z_far: float,
) -> np.ndarray:
    h_d, w_d = depth_m.shape
    h_c, w_c = color_shape
    ys, xs = np.meshgrid(np.arange(h_d), np.arange(w_d), indexing="ij")
    z = depth_m.reshape(-1)
    valid = np.isfinite(z) & (z > z_near) & (z < z_far)
    if not np.any(valid):
        return np.full((h_c, w_c), 0.0, dtype=np.float32)

    xs = xs.reshape(-1)[valid]
    ys = ys.reshape(-1)[valid]
    z = z[valid]
    x = (xs - k_depth[0, 2]) * z / k_depth[0, 0]
    y = (ys - k_depth[1, 2]) * z / k_depth[1, 1]
    points_depth = np.stack([x, y, z, np.ones_like(z)], axis=1)
    points_color = (t_depth_to_color @ points_depth.T).T[:, :3]
    zc = points_color[:, 2]
    valid_z = zc > z_near
    points_color = points_color[valid_z]
    zc = zc[valid_z]
    u = np.round(points_color[:, 0] * k_color[0, 0] / zc + k_color[0, 2]).astype(np.int32)
    v = np.round(points_color[:, 1] * k_color[1, 1] / zc + k_color[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < w_c) & (v >= 0) & (v < h_c)
    u = u[inside]
    v = v[inside]
    zc = zc[inside]

    aligned = np.full((h_c, w_c), np.inf, dtype=np.float32)
    for uu, vv, zz in zip(u, v, zc):
        if zz < aligned[vv, uu]:
            aligned[vv, uu] = zz
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned


def make_intrinsic_o3d(k: np.ndarray, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(k[0, 0]),
        float(k[1, 1]),
        float(k[0, 2]),
        float(k[1, 2]),
    )


def make_rgbd(color: np.ndarray, depth_m: np.ndarray, depth_scale: float = 1000.0) -> o3d.geometry.RGBDImage:
    color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    depth_u16 = np.clip(depth_m * depth_scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    depth_o3d = o3d.geometry.Image(depth_u16)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=float(depth_m[depth_m > 0].max()) if np.any(depth_m > 0) else 8.0,
        convert_rgb_to_intensity=False,
    )


def save_pose_plot(poses: List[np.ndarray], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not poses:
        return
    pts = np.stack([p[:3, 3] for p in poses], axis=0)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o")
    ax.set_title("Open3D Reconstruction Trajectory")
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
    parser.add_argument("--depth_dir", default="output/depth", type=str)
    parser.add_argument("--out_dir", default="output/reconstruction_v2", type=str)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--z_near", type=float, default=0.1)
    parser.add_argument("--z_far", type=float, default=8.0)
    parser.add_argument("--voxel_length", type=float, default=0.02)
    parser.add_argument("--sdf_trunc", type=float, default=0.06)
    parser.add_argument("--max_depth_diff", type=float, default=0.07)
    parser.add_argument("--min_success_fitness", type=float, default=0.15)
    args = parser.parse_args()

    set_logging()
    assert args.frame_stride >= 1

    data_dir = args.data_dir
    depth_dir = args.depth_dir if os.path.isabs(args.depth_dir) else os.path.join(data_dir, args.depth_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(data_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, "camera_intrinsics.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    k_depth = build_k(meta["pipeline_camera_param"]["depth_intrinsic"])
    k_color = build_k(meta["pipeline_camera_param"]["rgb_intrinsic"])
    t_depth_to_color = make_transform_matrix(
        meta["pipeline_camera_param"]["transform"]["rot"],
        meta["pipeline_camera_param"]["transform"]["transform"],
        translation_scale=0.001,
    )

    frame_rows = load_frame_rows(os.path.join(data_dir, "frame_timestamps.csv"))
    imu_samples = load_imu_samples(os.path.join(data_dir, "imu_data.csv"))
    color_files = list_indexed_files(os.path.join(data_dir, "color"), ".png")
    depth_files = list_indexed_files(depth_dir, ".npy")

    frame_indices = sorted(depth_files.keys())[:: args.frame_stride]
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise RuntimeError(f"No predicted depth files found in {depth_dir}")

    recon_frames: List[ReconFrame] = []
    pose_timestamps: List[int] = []
    for frame_index in frame_indices:
        row = frame_rows.get(frame_index)
        if row is None or row.pose_timestamp is None:
            logging.warning("Skip %06d: missing frame timestamps", frame_index)
            continue
        color_index = choose_color_frame(row, frame_rows, color_files)
        if color_index is None:
            logging.warning("Skip %06d: no matched color frame", frame_index)
            continue
        color = imageio.imread(color_files[color_index])
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=2)
        color = color[..., :3]
        pred_depth = np.load(depth_files[frame_index]).astype(np.float32)
        if pred_depth.ndim != 2:
            logging.warning("Skip %06d: invalid depth shape %s", frame_index, pred_depth.shape)
            continue
        depth_color = align_depth_to_color(
            pred_depth,
            k_depth,
            k_color,
            t_depth_to_color,
            color_shape=color.shape[:2],
            z_near=args.z_near,
            z_far=args.z_far,
        )
        recon_frames.append(
            ReconFrame(
                frame_index=frame_index,
                color_index=color_index,
                pose_timestamp=row.pose_timestamp,
                color=color,
                depth_color_m=depth_color,
            )
        )
        pose_timestamps.append(row.pose_timestamp)

    if not recon_frames:
        raise RuntimeError("No frames available for reconstruction.")

    imu_pose_map = integrate_imu_poses(imu_samples, pose_timestamps)
    color_intrinsic_o3d = make_intrinsic_o3d(k_color, recon_frames[0].color.shape[1], recon_frames[0].color.shape[0])
    odo_opt = o3d.pipelines.odometry.OdometryOption()
    if hasattr(odo_opt, "max_depth_diff"):
        odo_opt.max_depth_diff = args.max_depth_diff
    if hasattr(odo_opt, "depth_diff_max"):
        odo_opt.depth_diff_max = args.max_depth_diff

    rgbds: List[o3d.geometry.RGBDImage] = []
    for frame in recon_frames:
        rgbds.append(make_rgbd(frame.color, frame.depth_color_m))

    poses: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
    odom_infos: List[Dict[str, float]] = []
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    for i in range(1, len(recon_frames)):
        source = rgbds[i]
        target = rgbds[i - 1]
        init = np.eye(4, dtype=np.float64)
        imu_prev = imu_pose_map.get(recon_frames[i - 1].pose_timestamp)
        imu_curr = imu_pose_map.get(recon_frames[i].pose_timestamp)
        if imu_prev is not None and imu_curr is not None:
            init = np.linalg.inv(imu_prev) @ imu_curr
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            source,
            target,
            color_intrinsic_o3d,
            init,
            jacobian,
            odo_opt,
        )

        if success:
            global_pose = poses[-1] @ trans
            fitness_proxy = float(np.trace(info[:3, :3]) / 3.0)
            odom_infos.append(
                {
                    "frame_index": recon_frames[i].frame_index,
                    "color_index": recon_frames[i].color_index,
                    "odometry_success": True,
                    "information_trace": fitness_proxy,
                    "source": "open3d_rgbd_odometry",
                }
            )
        else:
            global_pose = poses[-1] @ init
            odom_infos.append(
                {
                    "frame_index": recon_frames[i].frame_index,
                    "color_index": recon_frames[i].color_index,
                    "odometry_success": False,
                    "information_trace": 0.0,
                    "source": "imu_fallback",
                }
            )
        poses.append(global_pose)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame, pose, rgbd in zip(recon_frames, poses, rgbds):
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, color_intrinsic_o3d, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    mesh_path = os.path.join(out_dir, "reconstruction_mesh.ply")
    pcd_path = os.path.join(out_dir, "reconstruction_pointcloud.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    o3d.io.write_point_cloud(pcd_path, pcd)

    pose_array = np.stack(poses, axis=0)
    np.save(os.path.join(out_dir, "poses.npy"), pose_array)
    save_pose_plot(poses, os.path.join(out_dir, "trajectory.png"))

    summary = {
        "data_dir": data_dir,
        "depth_dir": depth_dir,
        "num_frames_used": len(recon_frames),
        "mesh_path": mesh_path,
        "pointcloud_path": pcd_path,
        "voxel_length": args.voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "max_depth_diff": args.max_depth_diff,
        "z_near": args.z_near,
        "z_far": args.z_far,
        "depth_to_color_transform_m": t_depth_to_color[:3, 3].astype(float).tolist(),
        "pose_estimation": "Open3D RGBD odometry with IMU relative pose as initialization and fallback.",
        "frames": [
            {
                "frame_index": frame.frame_index,
                "color_index": frame.color_index,
                "pose_timestamp": frame.pose_timestamp,
                "pose_translation_m": poses[idx][:3, 3].astype(float).tolist(),
            }
            for idx, frame in enumerate(recon_frames)
        ],
        "odometry": odom_infos,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Saved Open3D mesh to %s", mesh_path)
    logging.info("Saved Open3D point cloud to %s", pcd_path)
    logging.info("Used %d frames", len(recon_frames))


if __name__ == "__main__":
    main()
