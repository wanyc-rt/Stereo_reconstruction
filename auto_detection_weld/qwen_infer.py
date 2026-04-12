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
    parser.add_argument("--raw_image_path", required=True)
    parser.add_argument("--overlay_image_path", required=True)
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
                f"{item['candidate_id']}[source={item.get('candidate_source', 'unknown')}]: bbox={item['bbox']}, area_px={item['area_px']}, "
                f"valid_depth_ratio={item['valid_depth_ratio']:.3f}, plane_inlier_ratio={item['plane_inlier_ratio']:.3f}, "
                f"border_touch_ratio={item['border_touch_ratio']:.3f}, center_distance_norm={item['center_distance_norm']:.3f}, "
                f"normal_z_abs={item['normal_z_abs']:.3f}, geometry_score={item['geometry_score']:.3f}, "
                f"sam_score={item.get('sam_score', 0.0):.3f}"
            )
        )
    pair_lines = []
    for item in payload.get("candidate_pairs", []):
        pair_lines.append(
            (
                f"{item['pair'][0]}-{item['pair'][1]}: near_band_area_px={item['near_band_area_px']}, "
                f"min_edge_gap_px={item.get('min_edge_gap_px')}, normal_alignment={item.get('normal_alignment')}, "
                f"band_linearity={item.get('band_linearity')}, band_elongation={item.get('band_elongation')}, "
                f"pair_source_support={item.get('pair_source_support')}, seam_shape_hint={item.get('seam_shape_hint')}, "
                f"pair_score={item.get('pair_score')}, iou={item.get('iou')}"
            )
        )
    prompt = (
        "你是一名焊缝定位助手。你会先看到原始 color 图，再看到叠加了候选区域标签的候选图。"
        "你必须先根据原始 color 图理解场景、钢板边界、焊缝纹理和反光干扰，再结合候选图完成候选筛选。"
        "不要跳过原始图像理解，也不要只依据标签框选结果。"
        "第二张图中叠加了若干候选区域，标签为 A/B/C/...。"
        "这些候选同时来自 SAM 分割和几何检测，source 字段会标明来源。"
        "你的任务分两步：先综合两类线索选出最可能构成焊接拼接关系的 plate candidate，再给出最适合作为焊缝搜索 ROI 的 roi candidate。"
        "必须使用焊接常识判断，不要只看面积。"
        "\n焊接常识规则："
        "\n1. 两块铁板彼此接近、边界之间只有窄带间隔时，该窄带最可能是拼接焊缝。"
        "\n2. 板材对接焊缝通常近似直线；如果工件有弧度，焊缝也应是平滑连续曲线，不应是蛇形、高频扭折或散乱斑块。"
        "\n3. 焊缝通常出现在两块板的接触边界附近，而不是某一整块钢板的内部孤立纹理、划痕、反光带。"
        "\n4. 地面、机械臂、底座、图像边缘碎片都不应作为钢板或焊缝 ROI。"
        "\n5. 若 candidate_pairs 中某一对的 pair_score 高、min_edge_gap_px 小、normal_alignment 高、band_linearity 高，则应优先把这对视为焊接候选对。"
        "\n5.1 如果同一空间区域同时被 SAM 和 geometry 支持，应提高可信度；如果两者明显冲突，优先保留更符合焊接边界关系和几何平面性的候选。"
        "\n6. plate candidate 的职责是表示构成焊接关系的钢板本体，优先选择平面性高、valid_depth_ratio 高、plane_inlier_ratio 高、border_touch_ratio 低的候选。"
        "\n7. roi candidate 的职责是限定焊缝搜索区域，优先选择贴近两块 plate 接缝窄带的候选；如果某个候选同时覆盖两块钢板上表面和中间窄带，允许只返回一个 roi_candidate_id。"
        "\n8. 如果 candidate_pairs 里的 seam_shape_hint 是 line，优先选择形成直而稳定窄带的 plate 对；如果 seam_shape_hint 是 curve，允许选择边界平滑弯曲但连续的 plate 对。"
        "\n9. 借鉴几何提取经验：直线焊缝看边界交线稳定性和线性，曲线焊缝看两表面交界的连续曲率与平滑连接，不要把局部反光、孔洞、刻痕当作 seam。"
        "\n选择偏好：plate 优先钢板上表面、优先中部、优先几何平面性高；roi 优先靠近接缝窄带、优先两板共同支持、优先直线或平滑曲线结构明显的候选。"
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
                {"type": "image", "image": args.raw_image_path},
                {"type": "text", "text": "第一张是原始 color 图。请先理解真实场景中的钢板边界、焊缝走向、阴影和反光。"},
                {"type": "image", "image": args.overlay_image_path},
                {"type": "text", "text": "第二张是候选叠加图。请在完成原图理解后，再利用候选标签和几何摘要筛选 plate candidate 与 roi candidate。"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[[args.raw_image_path, args.overlay_image_path]],
        return_tensors="pt",
    ).to(model.device)
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
