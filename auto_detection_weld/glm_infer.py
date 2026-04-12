import argparse
import base64
import json
import os
import re

from zai import ZhipuAiClient


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


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        raw = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{raw}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_image_path", required=True)
    parser.add_argument("--overlay_image_path", required=True)
    parser.add_argument("--candidates_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_key_env", default="ZAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("BIGMODEL_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} or BIGMODEL_API_KEY.")

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

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
        "\n规则："
        "\n1. plate candidate 表示钢板本体，优先选择平面性高、valid_depth_ratio 高、plane_inlier_ratio 高、border_touch_ratio 低的候选。"
        "\n2. roi candidate 表示焊缝搜索区域，优先选择贴近两块 plate 接缝窄带、同时得到 SAM 与 geometry 支持的候选。"
        "\n3. 当前任务更关注直线对接焊缝；若看到明显曲线，也必须确认其是连续平滑边界，而不是反光或纹理。"
        "\n4. 不要把地面、背景、边缘碎片、反光带、划痕当作钢板或焊缝。"
        "\n5. 如果 candidate_pairs 的 pair_score 高、min_edge_gap_px 小、normal_alignment 高，则优先考虑该 pair。"
        "\n请只输出 JSON，不要输出额外解释。格式严格为："
        "{\"plate_candidate_ids\":[\"A\",\"B\"],\"roi_candidate_ids\":[\"A\"],\"confidence\":0.0,\"reason\":\"简短中文理由\"}。"
        "\n用户任务：" + str(payload.get("prompt", "检测焊缝钢板区域")) +
        "\n候选摘要：\n" + "\n".join(candidate_lines) +
        "\n候选对摘要：\n" + "\n".join(pair_lines)
    )

    client = ZhipuAiClient(api_key=api_key)
    response = client.chat.completions.create(
        model=args.model_name,
        temperature=args.temperature,
        messages=[
            {
                "role": "system",
                "content": "你是一个严格输出 JSON 的工业视觉焊缝定位助手。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _encode_image(args.raw_image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": "第一张是原始 color 图。请先理解真实场景中的钢板边界、焊缝走向、阴影和反光。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _encode_image(args.overlay_image_path)
                        }
                    },
                    {
                        "type": "text",
                        "text": "第二张是候选叠加图。请在完成原图理解后，再利用候选标签和几何摘要筛选 plate candidate 与 roi candidate。"
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            },
        ],
    )
    answer = response.choices[0].message.content
    parsed = _extract_json(answer) or {
        "plate_candidate_ids": [],
        "roi_candidate_ids": [],
        "confidence": 0.0,
        "reason": str(answer).strip(),
        "raw_response": str(answer).strip(),
    }
    parsed.setdefault("backend", "glm_online")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
