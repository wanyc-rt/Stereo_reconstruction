from dataclasses import dataclass
from typing import Optional


@dataclass
class WeldDetectionConfig:
    prompt: str = "检测两块钢板，并提取中间焊接V字坡口，输出三维引导点"
    max_depth_m: float = 3.0
    min_depth_m: float = 0.1
    plane_distance_threshold_m: float = 0.008
    min_plane_points: int = 1500
    groove_band_width_px: int = 31
    groove_score_threshold: float = 0.45
    min_groove_area_px: int = 120
    pre_approach_offset_m: float = 0.08
    approach_offset_m: float = 0.02
    enable_open_vocab: bool = False
    detector_model: Optional[str] = None
    depth_source: str = "foundation_stereo"
    foundation_depth_dir: Optional[str] = None
    runtime_mode: str = "save_only"
    runtime_delay_ms: int = 30
    semantic_color_weight: float = 0.50
    geometry_normal_weight: float = 0.30
    geometry_depth_weight: float = 0.20
    roi_center_prior_weight: float = 0.35
    max_color_frame_delta: int = 2
    enable_sam: bool = False
    sam_python_executable: str = "/home/wycaihyj/Documents/CJB/FastSAM-Demo/backend/.venv/bin/python"
    sam_model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam_checkpoint_path: str = "/home/wycaihyj/Documents/CJB/FastSAM-Demo/backend/checkpoints/sam2.1_hiera_tiny.pt"
    sam_max_masks: int = 24
    sam_min_mask_area_px: int = 6000
    sam_top_k_candidates: int = 6
    enable_qwen: bool = False
    qwen_python_executable: str = "/home/wycaihyj/miniconda3/envs/python39/bin/python"
    qwen_model_path: str = "/home/wycaihyj/Documents/WYC/Data/Qwen2.5-vl-7B/2026"
    qwen_max_new_tokens: int = 256
