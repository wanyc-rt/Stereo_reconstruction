import argparse
import json
import re

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _extract_json(text: str):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--candidates_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    candidate_lines = []
    for item in payload.get("candidates", []):
        candidate_lines.append(
            (
                f"{item['candidate_id']}: bbox={item['bbox']}, area_px={item['area_px']}, "
                f"valid_depth_ratio={item['valid_depth_ratio']:.3f}, plane_inlier_ratio={item['plane_inlier_ratio']:.3f}, "
                f"border_touch_ratio={item['border_touch_ratio']:.3f}, center_distance_norm={item['center_distance_norm']:.3f}, "
                f"normal_z_abs={item['normal_z_abs']:.3f}, geometry_score={item['geometry_score']:.3f}"
            )
        )
    pair_lines = []
    for item in payload.get("candidate_pairs", []):
        pair_lines.append(
            (
                f"{item['pair'][0]}-{item['pair'][1]}: near_band_area_px={item['near_band_area_px']}, "
                f"min_edge_gap_px={item.get('min_edge_gap_px')}, normal_alignment={item.get('normal_alignment')}, "
                f"band_linearity={item.get('band_linearity')}, band_elongation={item.get('band_elongation')}, "
                f"pair_score={item.get('pair_score')}, iou={item.get('iou')}"
            )
        )
    prompt = (
        "你是一名焊缝定位助手。图中叠加了若干 SAM 候选区域，标签为 A/B/C/...。"
        "你的任务分两步：先选出最可能构成焊接拼接关系的钢板候选对，再给出最适合作为焊缝搜索 ROI 的候选。"
        "必须使用焊接常识判断，不要只看面积。"
        "\n焊接常识规则："
        "\n1. 两块铁板彼此接近、边界之间只有窄带间隔时，该窄带最可能是拼接焊缝。"
        "\n2. 板材对接焊缝通常近似直线；如果工件有弧度，焊缝也应是平滑连续曲线，不应是蛇形、高频扭折或散乱斑块。"
        "\n3. 焊缝通常出现在两块板的接触边界附近，而不是某一整块钢板的内部孤立纹理、划痕、反光带。"
        "\n4. 地面、机械臂、底座、图像边缘碎片都不应作为钢板或焊缝 ROI。"
        "\n5. 若 candidate_pairs 中某一对的 pair_score 高、min_edge_gap_px 小、normal_alignment 高、band_linearity 高，则应优先把这对视为焊接候选对。"
        "\n6. 如果两个 plate_candidate_ids 已经足够明确，roi_candidate_ids 应尽量选与该板对接缝窄带最相关的候选，而不是无关的第三块区域。"
        "\n7. 如果同一个候选已经覆盖两块钢板的上表面和中间窄带，允许只返回一个 roi_candidate_id。"
        "\n选择偏好：优先钢板上表面；优先中部；优先平面性高、边界接触率低、靠近接缝窄带、线性更强的候选。"
        "请只输出 JSON，不要输出额外解释。格式严格为："
        "{\"plate_candidate_ids\":[\"A\",\"B\"],\"roi_candidate_ids\":[\"A\"],\"confidence\":0.0,\"reason\":\"简短中文理由\"}。"
        "\n用户任务：" + str(payload.get("prompt", "检测焊缝钢板区域")) +
        "\n候选摘要：\n" + "\n".join(candidate_lines) +
        "\n候选对摘要：\n" + "\n".join(pair_lines)
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[args.image_path], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = response.split("assistant\n")[-1] if "assistant\n" in response else response
    parsed = _extract_json(answer) or {
        "plate_candidate_ids": [],
        "roi_candidate_ids": [],
        "confidence": 0.0,
        "reason": answer.strip(),
        "raw_response": answer.strip(),
    }
    parsed.setdefault("backend", "qwen_local")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
