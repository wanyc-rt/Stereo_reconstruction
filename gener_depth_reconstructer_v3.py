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


def list_raw_depth_files(folder: str) -> Dict[int, str]:
    files: Dict[int, str] = {}
    if not os.path.isdir(folder):
        return files
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".npy"):
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
    rgbd: o3d.geometry.RGBDImage
    pcd: o3d.geometry.PointCloud


@dataclass
class SegmentResult:
    segment_id: int
    start_frame_index: int
    end_frame_index: int
    num_frames: int
    mesh_path: str
    pointcloud_path: str
    poses_path: str
    trajectory_path: str
    frames: List[Dict[str, object]]
    edge_logs: List[Dict[str, object]]


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
    target_ts = row.color_timestamp if row.color_timestamp is not None else row.pose_timestamp
    if target_ts is None:
        return None
    candidates = []
    for idx, other in frame_rows.items():
        if idx not in color_files:
            continue
        ref_ts = other.color_timestamp if other.color_timestamp is not None else other.pose_timestamp
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
        return np.zeros((h_c, w_c), dtype=np.float32)
    xs = xs.reshape(-1)[valid]
    ys = ys.reshape(-1)[valid]
    z = z[valid]
    x = (xs - k_depth[0, 2]) * z / k_depth[0, 0]
    y = (ys - k_depth[1, 2]) * z / k_depth[1, 1]
    pts_depth = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_color = (t_depth_to_color @ pts_depth.T).T[:, :3]
    zc = pts_color[:, 2]
    valid_z = zc > z_near
    pts_color = pts_color[valid_z]
    zc = zc[valid_z]
    u = np.round(pts_color[:, 0] * k_color[0, 0] / zc + k_color[0, 2]).astype(np.int32)
    v = np.round(pts_color[:, 1] * k_color[1, 1] / zc + k_color[1, 2]).astype(np.int32)
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
    return o3d.camera.PinholeCameraIntrinsic(width, height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))


def make_rgbd(color: np.ndarray, depth_m: np.ndarray, depth_trunc: float) -> o3d.geometry.RGBDImage:
    depth_u16 = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color.astype(np.uint8)),
        o3d.geometry.Image(depth_u16),
        depth_scale=1000.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def make_point_cloud(rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, voxel_size: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > 10:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30))
    return pcd


def get_odometry_option(max_depth_diff: float):
    opt = o3d.pipelines.odometry.OdometryOption()
    if hasattr(opt, "max_depth_diff"):
        opt.max_depth_diff = max_depth_diff
    if hasattr(opt, "depth_diff_max"):
        opt.depth_diff_max = max_depth_diff
    return opt


def pairwise_registration(
    source_frame: ReconFrame,
    target_frame: ReconFrame,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    imu_init: np.ndarray,
    max_depth_diff: float,
    icp_distance: float,
) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, float]]:
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    odo_opt = get_odometry_option(max_depth_diff)

    trans_init = imu_init.copy()
    success_odo, trans_odo, info_odo = o3d.pipelines.odometry.compute_rgbd_odometry(
        source_frame.rgbd,
        target_frame.rgbd,
        intrinsic,
        trans_init,
        jacobian,
        odo_opt,
    )

    init_for_icp = trans_odo if success_odo else trans_init
    reg_icp = o3d.pipelines.registration.registration_icp(
        source_frame.pcd,
        target_frame.pcd,
        icp_distance,
        init_for_icp,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_frame.pcd,
        target_frame.pcd,
        icp_distance,
        reg_icp.transformation,
    )
    metrics = {
        "odometry_success": bool(success_odo),
        "odometry_info_trace": float(np.trace(info_odo)) if success_odo else 0.0,
        "icp_fitness": float(reg_icp.fitness),
        "icp_inlier_rmse": float(reg_icp.inlier_rmse),
        "icp_information_trace": float(np.trace(information)),
    }
    success = reg_icp.fitness > 0.08 and np.isfinite(reg_icp.inlier_rmse)
    return success, reg_icp.transformation, information, metrics


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
    ax.set_title("Pose Graph Optimized Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def reconstruct_segment(
    segment_id: int,
    segment_frames: List[ReconFrame],
    imu_pose_map: Dict[int, np.ndarray],
    color_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    out_dir: str,
    max_depth_diff: float,
    icp_distance: float,
    loop_interval: int,
    tsdf_voxel_length: float,
    sdf_trunc: float,
) -> SegmentResult:
    segment_dir = os.path.join(out_dir, f"segment_{segment_id:03d}")
    os.makedirs(segment_dir, exist_ok=True)

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    current_pose = np.eye(4, dtype=np.float64)
    edge_logs: List[Dict[str, object]] = []

    for i in range(1, len(segment_frames)):
        src = segment_frames[i]
        tgt = segment_frames[i - 1]
        imu_prev = imu_pose_map.get(tgt.pose_timestamp)
        imu_curr = imu_pose_map.get(src.pose_timestamp)
        imu_init = np.eye(4, dtype=np.float64)
        if imu_prev is not None and imu_curr is not None:
            imu_init = np.linalg.inv(imu_prev) @ imu_curr

        success, trans, info, metrics = pairwise_registration(
            src,
            tgt,
            color_intrinsic,
            imu_init,
            max_depth_diff,
            icp_distance,
        )
        if not success:
            raise RuntimeError(f"Segment {segment_id} has an invalid internal edge: {tgt.frame_index}->{src.frame_index}")

        current_pose = current_pose @ trans
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(current_pose)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i - 1,
                i,
                trans,
                info,
                uncertain=False,
            )
        )
        edge_logs.append(
            {
                "source_frame": tgt.frame_index,
                "target_frame": src.frame_index,
                "source": "rgbd_odometry_plus_icp",
                **metrics,
            }
        )

    if loop_interval > 1 and len(segment_frames) > loop_interval:
        for i in range(0, len(segment_frames) - loop_interval, loop_interval):
            j = i + loop_interval
            src = segment_frames[j]
            tgt = segment_frames[i]
            init = np.linalg.inv(pose_graph.nodes[i].pose) @ pose_graph.nodes[j].pose
            success, trans, info, metrics = pairwise_registration(
                src,
                tgt,
                color_intrinsic,
                init,
                max_depth_diff,
                icp_distance * 1.5,
            )
            if success and metrics["icp_fitness"] > 0.12:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i,
                        j,
                        trans,
                        info,
                        uncertain=True,
                    )
                )
                edge_logs.append(
                    {
                        "source_frame": tgt.frame_index,
                        "target_frame": src.frame_index,
                        "loop_edge": True,
                        "source": "rgbd_odometry_plus_icp",
                        **metrics,
                    }
                )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=icp_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )
    optimized_poses = [np.linalg.inv(node.pose) for node in pose_graph.nodes]

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame, pose in zip(segment_frames, optimized_poses):
        volume.integrate(frame.rgbd, color_intrinsic, np.linalg.inv(pose))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    mesh_path = os.path.join(segment_dir, "mesh.ply")
    pcd_path = os.path.join(segment_dir, "pointcloud.ply")
    poses_path = os.path.join(segment_dir, "poses.npy")
    trajectory_path = os.path.join(segment_dir, "trajectory.png")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    o3d.io.write_point_cloud(pcd_path, pcd)
    np.save(poses_path, np.stack(optimized_poses, axis=0))
    save_pose_plot(optimized_poses, trajectory_path)

    frames = [
        {
            "frame_index": frame.frame_index,
            "color_index": frame.color_index,
            "pose_timestamp": frame.pose_timestamp,
            "pose_translation_m": optimized_poses[idx][:3, 3].astype(float).tolist(),
        }
        for idx, frame in enumerate(segment_frames)
    ]
    return SegmentResult(
        segment_id=segment_id,
        start_frame_index=segment_frames[0].frame_index,
        end_frame_index=segment_frames[-1].frame_index,
        num_frames=len(segment_frames),
        mesh_path=mesh_path,
        pointcloud_path=pcd_path,
        poses_path=poses_path,
        trajectory_path=trajectory_path,
        frames=frames,
        edge_logs=edge_logs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/wycaihyj/Documents/WYC/Data/Foundaton_stereo/orbbec_recordings_20260404_160100", type=str)
    parser.add_argument("--depth_source", choices=["predicted", "raw"], default="predicted", type=str)
    parser.add_argument("--depth_dir", default="output/depth", type=str)
    parser.add_argument("--raw_depth_dir", default="depth", type=str)
    parser.add_argument("--out_dir", default="output/reconstruction_v3", type=str)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--z_near", type=float, default=0.1)
    parser.add_argument("--z_far", type=float, default=8.0)
    parser.add_argument("--pcd_voxel_size", type=float, default=0.03)
    parser.add_argument("--tsdf_voxel_length", type=float, default=0.02)
    parser.add_argument("--sdf_trunc", type=float, default=0.06)
    parser.add_argument("--max_depth_diff", type=float, default=0.07)
    parser.add_argument("--icp_distance", type=float, default=0.08)
    parser.add_argument("--loop_interval", type=int, default=5)
    parser.add_argument("--save_aligned_depth_preview", type=int, default=1)
    parser.add_argument("--raw_depth_scale_m", type=float, default=0.001)
    parser.add_argument("--min_segment_length", type=int, default=5)
    args = parser.parse_args()

    set_logging()
    assert args.frame_stride >= 1

    data_dir = args.data_dir
    pred_depth_dir = args.depth_dir if os.path.isabs(args.depth_dir) else os.path.join(data_dir, args.depth_dir)
    raw_depth_dir = args.raw_depth_dir if os.path.isabs(args.raw_depth_dir) else os.path.join(data_dir, args.raw_depth_dir)
    default_out = f"{args.out_dir}_{args.depth_source}" if args.out_dir.endswith("reconstruction_v3") else args.out_dir
    out_dir = default_out if os.path.isabs(default_out) else os.path.join(data_dir, default_out)
    os.makedirs(out_dir, exist_ok=True)
    preview_dir = os.path.join(out_dir, "aligned_depth_preview")
    if args.save_aligned_depth_preview:
        os.makedirs(preview_dir, exist_ok=True)

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
    if args.depth_source == "predicted":
        depth_files = list_indexed_files(pred_depth_dir, ".npy")
        active_depth_dir = pred_depth_dir
    else:
        depth_files = list_raw_depth_files(raw_depth_dir)
        active_depth_dir = raw_depth_dir

    frame_indices = sorted(depth_files.keys())[:: args.frame_stride]
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise RuntimeError(f"No depth files found in {active_depth_dir}")

    selected_rows = []
    pose_timestamps = []
    for frame_index in frame_indices:
        row = frame_rows.get(frame_index)
        if row is None or row.pose_timestamp is None:
            logging.warning("Skip %06d: missing frame row", frame_index)
            continue
        color_index = choose_color_frame(row, frame_rows, color_files)
        if color_index is None:
            logging.warning("Skip %06d: no matched color frame", frame_index)
            continue
        selected_rows.append((frame_index, row, color_index))
        pose_timestamps.append(row.pose_timestamp)

    if not selected_rows:
        raise RuntimeError("No frames could be aligned across depth/color/timestamps.")

    imu_pose_map = integrate_imu_poses(imu_samples, pose_timestamps)
    recon_frames: List[ReconFrame] = []
    color_intrinsic = None

    for frame_index, row, color_index in selected_rows:
        color = imageio.imread(color_files[color_index])
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=2)
        color = color[..., :3]
        pred_depth = np.load(depth_files[frame_index]).astype(np.float32)
        if args.depth_source == "raw":
            pred_depth = pred_depth * args.raw_depth_scale_m
        depth_color = align_depth_to_color(
            pred_depth,
            k_depth,
            k_color,
            t_depth_to_color,
            color.shape[:2],
            args.z_near,
            args.z_far,
        )
        if color_intrinsic is None:
            color_intrinsic = make_intrinsic_o3d(k_color, color.shape[1], color.shape[0])
        rgbd = make_rgbd(color, depth_color, args.z_far)
        pcd = make_point_cloud(rgbd, color_intrinsic, args.pcd_voxel_size)
        if len(pcd.points) < 100:
            logging.warning("Skip %06d: too few valid points after depth alignment", frame_index)
            continue
        recon_frames.append(
            ReconFrame(
                frame_index=frame_index,
                color_index=color_index,
                pose_timestamp=row.pose_timestamp,
                color=color,
                depth_color_m=depth_color,
                rgbd=rgbd,
                pcd=pcd,
            )
        )
        if args.save_aligned_depth_preview:
            vis = np.clip(depth_color / max(args.z_far, 1e-6), 0, 1)
            vis = (255 * vis).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(preview_dir, f"{frame_index:06d}.png"), vis)

    if len(recon_frames) < 2:
        raise RuntimeError("Not enough valid frames for V3 reconstruction.")

    segments: List[List[ReconFrame]] = []
    current_segment: List[ReconFrame] = [recon_frames[0]]
    transition_logs: List[Dict[str, object]] = []
    for i in range(1, len(recon_frames)):
        src = recon_frames[i]
        tgt = recon_frames[i - 1]
        imu_prev = imu_pose_map.get(tgt.pose_timestamp)
        imu_curr = imu_pose_map.get(src.pose_timestamp)
        imu_init = np.eye(4, dtype=np.float64)
        if imu_prev is not None and imu_curr is not None:
            imu_init = np.linalg.inv(imu_prev) @ imu_curr
        success, _, _, metrics = pairwise_registration(
            src,
            tgt,
            color_intrinsic,
            imu_init,
            args.max_depth_diff,
            args.icp_distance,
        )
        transition_logs.append(
            {
                "source_frame": tgt.frame_index,
                "target_frame": src.frame_index,
                "accepted_for_same_segment": bool(success),
                **metrics,
            }
        )
        if success:
            current_segment.append(src)
        else:
            if len(current_segment) >= args.min_segment_length:
                segments.append(current_segment)
            current_segment = [src]
    if len(current_segment) >= args.min_segment_length:
        segments.append(current_segment)

    segment_results: List[SegmentResult] = []
    for segment_id, segment_frames in enumerate(segments):
        result = reconstruct_segment(
            segment_id=segment_id,
            segment_frames=segment_frames,
            imu_pose_map=imu_pose_map,
            color_intrinsic=color_intrinsic,
            out_dir=out_dir,
            max_depth_diff=args.max_depth_diff,
            icp_distance=args.icp_distance,
            loop_interval=args.loop_interval,
            tsdf_voxel_length=args.tsdf_voxel_length,
            sdf_trunc=args.sdf_trunc,
        )
        segment_results.append(result)

    summary = {
        "data_dir": data_dir,
        "depth_source": args.depth_source,
        "depth_dir": active_depth_dir,
        "num_frames_used": len(recon_frames),
        "num_segments": len(segment_results),
        "min_segment_length": args.min_segment_length,
        "pcd_voxel_size": args.pcd_voxel_size,
        "tsdf_voxel_length": args.tsdf_voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "max_depth_diff": args.max_depth_diff,
        "icp_distance": args.icp_distance,
        "loop_interval": args.loop_interval,
        "z_near": args.z_near,
        "z_far": args.z_far,
        "pose_estimation": "Segmented reconstruction: only consecutive frames with successful RGBD odometry + ICP are grouped; each segment runs local pose graph optimization and TSDF fusion.",
        "segments": [
            {
                "segment_id": result.segment_id,
                "start_frame_index": result.start_frame_index,
                "end_frame_index": result.end_frame_index,
                "num_frames": result.num_frames,
                "mesh_path": result.mesh_path,
                "pointcloud_path": result.pointcloud_path,
                "poses_path": result.poses_path,
                "trajectory_path": result.trajectory_path,
                "frames": result.frames,
                "edge_logs": result.edge_logs,
            }
            for result in segment_results
        ],
        "transition_logs": transition_logs,
    }
    with open(os.path.join(out_dir, f"summary_v3_{args.depth_source}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(
        "Built %d stable segments from %d frames using %s depth",
        len(segment_results),
        len(recon_frames),
        args.depth_source,
    )


if __name__ == "__main__":
    main()
