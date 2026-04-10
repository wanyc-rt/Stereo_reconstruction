import argparse
import json

import numpy as np
import torch
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--max_masks", type=int, default=24)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image_path).convert("RGB"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.model_cfg, args.checkpoint_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.90,
        min_mask_region_area=256,
    )
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=lambda item: float(item.get("area", 0)), reverse=True)[: args.max_masks]

    packed_masks = []
    meta = {"masks": []}
    for item in masks:
        mask = item["segmentation"].astype(np.uint8)
        packed_masks.append(mask)
        bbox_xywh = item.get("bbox", [0, 0, 0, 0])
        x0, y0, bw, bh = [int(v) for v in bbox_xywh]
        meta["masks"].append(
            {
                "bbox": [x0, y0, x0 + bw - 1, y0 + bh - 1],
                "area": int(item.get("area", int(mask.sum()))),
                "predicted_iou": float(item.get("predicted_iou", 0.0)),
                "stability_score": float(item.get("stability_score", 0.0)),
            }
        )
    if packed_masks:
        np.savez_compressed(args.output_npz, masks=np.stack(packed_masks, axis=0))
    else:
        np.savez_compressed(args.output_npz, masks=np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8))
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
