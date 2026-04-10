import argparse
import os
import sys

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
if 'QT_QPA_FONTDIR' not in os.environ:
    os.environ['QT_QPA_FONTDIR'] = FONT_DIR
if 'OPENCV_QT_FONTDIR' not in os.environ:
    os.environ['OPENCV_QT_FONTDIR'] = FONT_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auto_detection_weld.config import WeldDetectionConfig
from auto_detection_weld.pipeline import WeldDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='检测两块钢板，并提取中间焊接V字坡口，输出三维引导点')
    parser.add_argument('--frame_id', type=str, default=None)
    parser.add_argument('--start_frame_id', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=-1)
    parser.add_argument('--max_depth_m', type=float, default=3.0)
    parser.add_argument('--min_depth_m', type=float, default=0.1)
    parser.add_argument('--enable_open_vocab', type=int, default=0)
    parser.add_argument('--detector_model', type=str, default=None)
    parser.add_argument('--depth_source', type=str, default='foundation_stereo', choices=['foundation_stereo', 'raw', 'auto'])
    parser.add_argument('--foundation_depth_dir', type=str, default=None)
    parser.add_argument('--runtime_mode', type=str, default='save_only', choices=['popup', 'save_only', 'auto'])
    parser.add_argument('--runtime_delay_ms', type=int, default=30)
    parser.add_argument('--semantic_color_weight', type=float, default=0.50)
    parser.add_argument('--geometry_normal_weight', type=float, default=0.30)
    parser.add_argument('--geometry_depth_weight', type=float, default=0.20)
    parser.add_argument('--roi_center_prior_weight', type=float, default=0.35)
    parser.add_argument('--max_color_frame_delta', type=int, default=2)
    parser.add_argument('--enable_sam', type=int, default=0)
    parser.add_argument('--sam_python_executable', type=str, default="/home/wycaihyj/Documents/CJB/FastSAM-Demo/backend/.venv/bin/python")
    parser.add_argument('--sam_model_cfg', type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument('--sam_checkpoint_path', type=str, default="/home/wycaihyj/Documents/CJB/FastSAM-Demo/backend/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument('--sam_max_masks', type=int, default=24)
    parser.add_argument('--sam_min_mask_area_px', type=int, default=6000)
    parser.add_argument('--sam_top_k_candidates', type=int, default=6)
    parser.add_argument('--enable_qwen', type=int, default=0)
    parser.add_argument('--qwen_python_executable', type=str, default="/home/wycaihyj/miniconda3/envs/python39/bin/python")
    parser.add_argument('--qwen_model_path', type=str, default="/home/wycaihyj/Documents/WYC/Data/Qwen2.5-vl-7B/2026")
    parser.add_argument('--qwen_max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    cfg = WeldDetectionConfig(
        prompt=args.prompt,
        max_depth_m=args.max_depth_m,
        min_depth_m=args.min_depth_m,
        enable_open_vocab=bool(args.enable_open_vocab),
        detector_model=args.detector_model,
        depth_source=args.depth_source,
        foundation_depth_dir=args.foundation_depth_dir,
        runtime_mode=args.runtime_mode,
        runtime_delay_ms=args.runtime_delay_ms,
        semantic_color_weight=args.semantic_color_weight,
        geometry_normal_weight=args.geometry_normal_weight,
        geometry_depth_weight=args.geometry_depth_weight,
        roi_center_prior_weight=args.roi_center_prior_weight,
        max_color_frame_delta=args.max_color_frame_delta,
        enable_sam=bool(args.enable_sam),
        sam_python_executable=args.sam_python_executable,
        sam_model_cfg=args.sam_model_cfg,
        sam_checkpoint_path=args.sam_checkpoint_path,
        sam_max_masks=args.sam_max_masks,
        sam_min_mask_area_px=args.sam_min_mask_area_px,
        sam_top_k_candidates=args.sam_top_k_candidates,
        enable_qwen=bool(args.enable_qwen),
        qwen_python_executable=args.qwen_python_executable,
        qwen_model_path=args.qwen_model_path,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
    )
    pipeline = WeldDetectionPipeline(cfg)
    pipeline.run_sequence(
        args.data_dir,
        args.output_dir,
        frame_id=args.frame_id,
        start_frame_id=args.start_frame_id,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
