import os
from typing import Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

from .config import WeldDetectionConfig
from .detector import HeuristicPlateDetector, OpenVocabularyDetector
from .geometry import depth_to_xyz, extract_groove_from_roi, generate_waypoints
from .hybrid import (
    QwenReasoner,
    SAMSegmenter,
    build_joint_roi_mask,
    build_segment_candidates,
    resolve_reasoned_candidates,
    select_geometry_candidates,
    serialize_reasoning_for_json,
    summarize_candidate_pairs,
)
from .io_utils import (
    collect_frames,
    copy_file_if_exists,
    draw_polyline,
    ensure_dir,
    extract_camera_alignment,
    load_camera_meta,
    load_color,
    load_depth,
    make_alignment_overlay,
    make_depth_vis,
    make_normal_vis,
    overlay_mask,
    project_points_to_color,
    render_color_aligned_to_depth,
    save_json,
    save_colored_point_cloud,
)


class WeldDetectionPipeline:
    def __init__(self, config: WeldDetectionConfig) -> None:
        self.config = config
        self.heuristic_detector = HeuristicPlateDetector(
            plane_distance_threshold_m=config.plane_distance_threshold_m,
            min_plane_points=config.min_plane_points,
        )
        self.open_vocab_detector = OpenVocabularyDetector(config.detector_model) if config.enable_open_vocab else None
        self.sam_segmenter = SAMSegmenter(
            python_executable=config.sam_python_executable,
            model_cfg=config.sam_model_cfg,
            checkpoint_path=config.sam_checkpoint_path,
            max_masks=config.sam_max_masks,
        ) if config.enable_sam else None
        self.qwen_reasoner = QwenReasoner(
            python_executable=config.qwen_python_executable,
            model_path=config.qwen_model_path,
            max_new_tokens=config.qwen_max_new_tokens,
        ) if config.enable_qwen else None
        self._runtime_window_name = 'weld_runtime'
        self._runtime_popup_enabled = self._resolve_runtime_popup_enabled()


    def _resolve_runtime_popup_enabled(self) -> bool:
        mode = self.config.runtime_mode
        if mode == 'save_only':
            return False
        if mode == 'popup':
            return True
        return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

    def _select_detector(self):
        if self.open_vocab_detector is not None and any(
            keyword in self.config.prompt.lower() for keyword in ['steel', 'plate', 'groove', 'weld', '钢板', '坡口', '焊缝']
        ):
            return self.open_vocab_detector
        return self.heuristic_detector

    def _make_fallback_color(self, depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        vis = np.zeros(depth_m.shape, dtype=np.uint8)
        if np.any(valid):
            vals = depth_m[valid]
            lo = float(np.percentile(vals, 5))
            hi = float(np.percentile(vals, 95))
            scale = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            vis = (scale * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)[:, :, ::-1]

    def _show_runtime(self, frame_id: str, images: Dict[str, np.ndarray], hold: bool = False) -> None:
        if not self._runtime_popup_enabled:
            return
        try:
            labeled = []
            for name, img in images.items():
                canvas = img.copy()
                if canvas.ndim == 2:
                    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
                if canvas.dtype != np.uint8:
                    canvas = np.clip(canvas * 255 if canvas.max() <= 1.0 else canvas, 0, 255).astype(np.uint8)
                cv2.putText(canvas, f'{frame_id}: {name}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                labeled.append(canvas)
            if not labeled:
                return
            h = min(img.shape[0] for img in labeled)
            resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_LINEAR) for img in labeled]
            montage = np.concatenate(resized, axis=1)
            cv2.imshow(self._runtime_window_name, montage[:, :, ::-1])
            if hold:
                cv2.waitKey(0)
            else:
                cv2.waitKey(max(self.config.runtime_delay_ms, 1))
        except Exception:
            pass

    def _write_debug_failure(self, output_dir: str, frame_id: str, color: np.ndarray, reason: str, extra: Optional[dict] = None) -> None:
        debug_dir = os.path.join(output_dir, 'debug')
        ensure_dir(debug_dir)
        canvas = color.copy()
        cv2.putText(canvas, reason, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        imageio.imwrite(os.path.join(debug_dir, f'{frame_id}_debug.png'), canvas)
        payload = {'frame_id': frame_id, 'status': 'failed', 'reason': reason}
        if extra:
            payload.update(extra)
        save_json(os.path.join(debug_dir, f'{frame_id}_debug.json'), payload)

    def _detect_with_sam(self, color: np.ndarray, xyz_map: np.ndarray, valid: np.ndarray, output_dir: str, frame_id: str) -> Optional[Dict[str, object]]:
        if self.sam_segmenter is None or not self.sam_segmenter.is_available():
            return None
        sam_dir = os.path.join(output_dir, "sam")
        ensure_dir(sam_dir)
        segment_masks = self.sam_segmenter.generate(color, sam_dir, frame_id)
        if not segment_masks:
            return None
        candidates = build_segment_candidates(
            segment_masks=segment_masks,
            xyz_map=xyz_map,
            valid_mask=valid,
            min_mask_area_px=self.config.sam_min_mask_area_px,
            plane_distance_threshold_m=self.config.plane_distance_threshold_m,
            min_plane_points=self.config.min_plane_points,
        )
        if not candidates:
            return None
        top_candidates = candidates[: self.config.sam_top_k_candidates]
        candidate_pairs = summarize_candidate_pairs(top_candidates)
        reasoning = self.qwen_reasoner.select(color, top_candidates, candidate_pairs, sam_dir, frame_id, self.config.prompt) if self.qwen_reasoner else None
        selected = resolve_reasoned_candidates(top_candidates, reasoning, top_k=2)
        if not selected:
            selected = select_geometry_candidates(top_candidates, top_k=2)
        roi_mask, roi_meta = build_joint_roi_mask(selected)
        plates: List[Dict[str, object]] = []
        for candidate in selected:
            plates.append(
                {
                    "mask": candidate.mask,
                    "bbox": candidate.bbox,
                    "candidate_id": candidate.candidate_id,
                    "geometry_score": candidate.geometry_score,
                    "plane_normal": candidate.plane_normal,
                }
            )
        return {
            "plates": plates,
            "roi_mask": roi_mask,
            "supports_single_plate_roi": True,
            "sam_candidates": [candidate.to_dict() for candidate in top_candidates],
            "candidate_pairs": candidate_pairs,
            "reasoning": serialize_reasoning_for_json(reasoning),
            "roi_construction": roi_meta,
            "backend": "sam_qwen" if reasoning is not None else "sam_geometry",
        }

    def _save_input_artifacts(self, frame: Dict[str, str], color: np.ndarray, depth_m: np.ndarray, output_dir: str) -> Dict[str, Optional[str]]:
        input_dir = os.path.join(output_dir, 'inputs')
        color_dir = os.path.join(input_dir, 'color')
        depth_dir = os.path.join(input_dir, 'depth')
        cloud_dir = os.path.join(input_dir, 'cloud')
        ensure_dir(color_dir)
        ensure_dir(depth_dir)
        ensure_dir(cloud_dir)

        frame_id = frame['frame_id']
        color_copy_path = os.path.join(color_dir, f'{frame_id}_color.png')
        depth_raw_copy_path = os.path.join(depth_dir, f'{frame_id}_depth_source{os.path.splitext(frame["depth_path"])[1]}')
        depth_vis_path = os.path.join(depth_dir, f'{frame_id}_depth_vis.png')

        color_saved = copy_file_if_exists(frame.get('color_path'), color_copy_path)
        if not color_saved:
            imageio.imwrite(color_copy_path, color)
        copy_file_if_exists(frame.get('depth_path'), depth_raw_copy_path)
        imageio.imwrite(depth_vis_path, make_depth_vis(depth_m))

        cloud_src = None
        denoise_src = None
        if frame.get('depth_source') == 'foundation_stereo' and frame.get('foundation_root'):
            cloud_root = os.path.join(frame['foundation_root'], 'cloud')
            candidate = os.path.join(cloud_root, f'{frame_id}_pred_cloud.ply')
            candidate_denoise = os.path.join(cloud_root, f'{frame_id}_pred_cloud_denoise.ply')
            if os.path.exists(candidate):
                cloud_src = candidate
                copy_file_if_exists(candidate, os.path.join(cloud_dir, os.path.basename(candidate)))
            if os.path.exists(candidate_denoise):
                denoise_src = candidate_denoise
                copy_file_if_exists(candidate_denoise, os.path.join(cloud_dir, os.path.basename(candidate_denoise)))

        return {
            'saved_color_path': color_copy_path,
            'saved_depth_vis_path': depth_vis_path,
            'saved_depth_source_path': depth_raw_copy_path if os.path.exists(depth_raw_copy_path) else None,
            'saved_pointcloud_path': os.path.join(cloud_dir, os.path.basename(cloud_src)) if cloud_src else None,
            'saved_pointcloud_denoise_path': os.path.join(cloud_dir, os.path.basename(denoise_src)) if denoise_src else None,
        }

    def run_frame(self, frame: Dict[str, str], camera_meta: dict, k_depth: np.ndarray, output_dir: str) -> Optional[Dict[str, object]]:
        depth_m = load_depth(frame['depth_path'])
        valid = np.isfinite(depth_m) & (depth_m > self.config.min_depth_m) & (depth_m < self.config.max_depth_m)
        xyz_map = depth_to_xyz(depth_m, k_depth)
        raw_color = load_color(frame.get('color_path'))
        alignment = extract_camera_alignment(camera_meta)
        aligned_color = None
        aligned_coverage = None
        if raw_color is not None and alignment.get('k_color') is not None and alignment.get('t_depth_to_color') is not None:
            aligned_color, aligned_coverage = render_color_aligned_to_depth(
                xyz_depth=xyz_map,
                color_img=raw_color,
                k_color=alignment['k_color'],
                t_depth_to_color=alignment['t_depth_to_color'],
                valid_mask=valid,
            )

        color = aligned_color
        if color is None or color.shape[:2] != depth_m.shape[:2] or (aligned_coverage is not None and int(aligned_coverage.sum()) == 0):
            color = raw_color
            if color is None:
                color = self._make_fallback_color(depth_m)
            if color.shape[:2] != depth_m.shape[:2]:
                color = cv2.resize(color, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_LINEAR)
            aligned_coverage = np.any(color > 0, axis=2)

        input_artifacts = self._save_input_artifacts(frame, color, depth_m, output_dir)
        if valid.sum() < self.config.min_plane_points:
            self._write_debug_failure(output_dir, frame['frame_id'], color, 'not_enough_valid_depth', {'valid_points': int(valid.sum())})
            return None

        detect_result = None
        if frame.get('color_path') and self.sam_segmenter is not None:
            detect_result = self._detect_with_sam(color, xyz_map, valid, output_dir, frame['frame_id'])
        if detect_result is None:
            detector = self._select_detector()
            detect_result = detector.detect(color, depth_m, xyz_map, valid)
        plates = detect_result.get('plates', [])
        roi_mask = detect_result.get('roi_mask')
        min_required_plates = 1 if detect_result.get('supports_single_plate_roi') else 2
        if len(plates) < min_required_plates or roi_mask is None:
            heuristic_result = self.heuristic_detector.detect(color, depth_m, xyz_map, valid)
            plates = heuristic_result.get('plates', [])
            roi_mask = heuristic_result.get('roi_mask')
            if len(plates) < 2 or roi_mask is None:
                self._write_debug_failure(output_dir, frame['frame_id'], color, 'plate_detection_failed', {'num_plates': len(plates)})
                return None

        groove = extract_groove_from_roi(
            color=color,
            depth_m=depth_m,
            xyz_map=xyz_map,
            k_depth=k_depth,
            roi_mask=roi_mask,
            prompt=self.config.prompt,
            groove_score_threshold=self.config.groove_score_threshold,
            min_groove_area_px=self.config.min_groove_area_px,
            semantic_color_weight=self.config.semantic_color_weight,
            geometry_normal_weight=self.config.geometry_normal_weight,
            geometry_depth_weight=self.config.geometry_depth_weight,
            roi_center_prior_weight=self.config.roi_center_prior_weight,
        )
        if groove is None:
            debug_vis = color.copy()
            if len(plates) >= 1 and 'mask' in plates[0]:
                debug_vis = overlay_mask(debug_vis, plates[0]['mask'], (255, 0, 0))
            if len(plates) >= 2 and 'mask' in plates[1]:
                debug_vis = overlay_mask(debug_vis, plates[1]['mask'], (0, 0, 255))
            self._write_debug_failure(
                output_dir,
                frame['frame_id'],
                debug_vis,
                'groove_extraction_failed',
                {
                    'color_frame_id': frame.get('color_frame_id'),
                    'depth_source': frame.get('depth_source'),
                },
            )
            return None

        waypoints = generate_waypoints(
            target_point=groove['target_point_m'],
            approach_dir=groove['approach_direction_camera'],
            start=groove['weld_start_point_m'],
            end=groove['weld_end_point_m'],
            pre_offset=self.config.pre_approach_offset_m,
            approach_offset=self.config.approach_offset_m,
        )

        vis = color.copy()
        if len(plates) >= 1 and 'mask' in plates[0]:
            vis = overlay_mask(vis, plates[0]['mask'], (255, 0, 0))
        if len(plates) >= 2 and 'mask' in plates[1]:
            vis = overlay_mask(vis, plates[1]['mask'], (0, 0, 255))
        vis = overlay_mask(vis, groove['groove_mask'], (255, 180, 0))
        vis = draw_polyline(vis, groove['centerline_pixels'], (0, 255, 0), 2)
        target_uv = np.round(groove['centerline_pixels'][len(groove['centerline_pixels']) // 2]).astype(np.int32)
        cv2.circle(vis, tuple(target_uv.tolist()), 6, (255, 255, 0), -1, cv2.LINE_AA)

        score_vis = cv2.applyColorMap((np.clip(groove['score_map'], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        normal_vis = cv2.applyColorMap((np.clip(groove['normal_change'], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
        normal_rgb_vis = make_normal_vis(groove['normals'], valid_mask=valid) if 'normals' in groove else np.zeros_like(color)
        self._show_runtime(frame['frame_id'], {'color': color, 'score': score_vis, 'normal_change': normal_vis, 'result': vis})

        vis_dir = os.path.join(output_dir, 'vis')
        json_dir = os.path.join(output_dir, 'json')
        cloud_dir = os.path.join(output_dir, 'cloud')
        ensure_dir(vis_dir)
        ensure_dir(json_dir)
        ensure_dir(cloud_dir)
        imageio.imwrite(os.path.join(vis_dir, f"{frame['frame_id']}_weld_detection.png"), vis)
        imageio.imwrite(os.path.join(vis_dir, f"{frame['frame_id']}_score.png"), score_vis)
        imageio.imwrite(os.path.join(vis_dir, f"{frame['frame_id']}_normal_change.png"), normal_vis)
        imageio.imwrite(os.path.join(vis_dir, f"{frame['frame_id']}_normal_rgb.png"), normal_rgb_vis)

        point_cloud_path = None
        aligned_color_path = os.path.join(vis_dir, f"{frame['frame_id']}_color_aligned_to_depth.png")
        alignment_overlay_path = os.path.join(vis_dir, f"{frame['frame_id']}_color_depth_alignment_overlay.png")
        imageio.imwrite(aligned_color_path, color)
        imageio.imwrite(alignment_overlay_path, make_alignment_overlay(color, depth_m, aligned_coverage))
        if alignment.get('k_color') is not None and alignment.get('t_depth_to_color') is not None:
            point_colors = project_points_to_color(
                xyz_depth=xyz_map,
                color_img=raw_color if raw_color is not None else color,
                k_color=alignment['k_color'],
                t_depth_to_color=alignment['t_depth_to_color'],
            )
            point_cloud_path = os.path.join(cloud_dir, f"{frame['frame_id']}_colorized_depth_cloud.ply")
            save_colored_point_cloud(xyz_map, point_colors, point_cloud_path, valid_mask=valid)
        else:
            point_colors = color.reshape(-1, 3)
            point_cloud_path = os.path.join(cloud_dir, f"{frame['frame_id']}_aligned_color_depth_cloud.ply")
            save_colored_point_cloud(xyz_map, point_colors, point_cloud_path, valid_mask=valid)

        result = {
            'frame_id': frame['frame_id'],
            'color_frame_id': frame.get('color_frame_id'),
            'prompt': self.config.prompt,
            'depth_source': frame.get('depth_source'),
            'depth_path': frame.get('depth_path'),
            'input_artifacts': input_artifacts,
            'plate_a_bbox': plates[0].get('bbox', []) if len(plates) >= 1 else [],
            'plate_b_bbox': plates[1].get('bbox', []) if len(plates) >= 2 else [],
            'target_point_m': groove['target_point_m'].tolist(),
            'approach_direction_camera': groove['approach_direction_camera'].tolist(),
            'seam_direction_camera': groove['seam_direction_camera'].tolist(),
            'weld_start_point_m': groove['weld_start_point_m'].tolist(),
            'weld_end_point_m': groove['weld_end_point_m'].tolist(),
            'centerline_pixels': groove['centerline_pixels'].tolist(),
            'centerline_straightness': float(groove.get('centerline_straightness', 0.0)),
            'centerline_smoothness': float(groove.get('centerline_smoothness', 0.0)),
            'waypoints': {k: v.tolist() for k, v in waypoints.items()},
            'detector_backend': detect_result.get('backend', 'heuristic'),
            'sam_candidates': detect_result.get('sam_candidates'),
            'candidate_pairs': detect_result.get('candidate_pairs'),
            'reasoning': detect_result.get('reasoning'),
            'roi_construction': detect_result.get('roi_construction'),
            'saved_normal_rgb_path': os.path.join(vis_dir, f"{frame['frame_id']}_normal_rgb.png"),
            'saved_normal_change_path': os.path.join(vis_dir, f"{frame['frame_id']}_normal_change.png"),
            'saved_score_path': os.path.join(vis_dir, f"{frame['frame_id']}_score.png"),
            'saved_color_aligned_path': aligned_color_path,
            'saved_color_depth_overlay_path': alignment_overlay_path,
            'saved_colorized_depth_cloud_path': point_cloud_path,
            'color_depth_alignment': {
                'used_depth_intrinsic': alignment.get('k_depth').tolist() if alignment.get('k_depth') is not None else None,
                'used_color_intrinsic': alignment.get('k_color').tolist() if alignment.get('k_color') is not None else None,
                'depth_to_color_transform': alignment.get('t_depth_to_color').tolist() if alignment.get('t_depth_to_color') is not None else None,
                'color_frame_was_nearest_match': frame.get('color_frame_id') != frame.get('frame_id'),
                'aligned_color_coverage_ratio': float(aligned_coverage.mean()) if aligned_coverage is not None else None,
            },
        }
        save_json(os.path.join(json_dir, f"{frame['frame_id']}_weld_detection.json"), result)
        return result

    def run_sequence(
        self,
        data_dir: str,
        output_dir: str,
        frame_id: Optional[str] = None,
        start_frame_id: Optional[str] = None,
        max_frames: int = -1,
    ) -> List[Dict[str, object]]:
        ensure_dir(output_dir)
        camera_meta, k_depth = load_camera_meta(data_dir)
        frames = collect_frames(
            data_dir,
            frame_id=frame_id,
            start_frame_id=start_frame_id,
            max_frames=max_frames,
            depth_source=self.config.depth_source,
            foundation_depth_dir=self.config.foundation_depth_dir,
            max_color_frame_delta=self.config.max_color_frame_delta,
        )
        results: List[Dict[str, object]] = []
        for idx, frame in enumerate(frames):
            result = self.run_frame(frame, camera_meta, k_depth, output_dir)
            if result is not None:
                results.append(result)
            if self._runtime_popup_enabled and idx == len(frames) - 1:
                try:
                    cv2.waitKey(0)
                except Exception:
                    pass
        if self._runtime_popup_enabled:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        save_json(os.path.join(output_dir, 'summary.json'), {'num_results': len(results), 'num_frames': len(frames), 'depth_source': self.config.depth_source, 'results': results})
        return results
