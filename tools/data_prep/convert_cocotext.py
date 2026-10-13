#!/usr/bin/env python3
"""Convert COCO-Text v2.0 (cocotext.v2.json) to COCO detection format, 1 class "text".

Writes into <out-dir>:
  cocotext_train.json      official train split (all of it, for a final retrain)
  cocotext_train_dev.json  train minus the dev images (training during development)
  cocotext_dev.json        held-out part of train (model selection)
  cocotext_val.json        official val split (final evaluation only)

Images are the COCO train2014 images (COCO_train2014_*.jpg), so set
train_img_dir = val_img_dir = test_img_dir = "train2014" in the exp.

  python tools/data_prep/convert_cocotext.py --src cocotext.v2.json \
      --out-dir datasets/COCOText/annotations --dev-images 2000
"""
import argparse
import json
import os

from make_dev_split import split_coco


def convert(src, split, legible_only=False, keep_empty=False):
    images, annotations = [], []
    for img in src["imgs"].values():
        if img["set"] != split:
            continue
        if not keep_empty and not src["imgToAnns"].get(str(img["id"])):
            continue
        images.append({
            "id": int(img["id"]), "file_name": img["file_name"],
            "width": img["width"], "height": img["height"],
        })
        for aid in src["imgToAnns"].get(str(img["id"]), []):
            a = src["anns"][str(aid)]
            if legible_only and a.get("legibility") != "legible":
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": int(a["id"]), "image_id": int(img["id"]), "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                "legibility": a.get("legibility"), "language": a.get("language"),
                "text_class": a.get("class"),
            })
    return {
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "text", "supercategory": "text"}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cocotext.v2.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-images", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--legible-only", action="store_true")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also keep images without text (default: only images with text)")
    args = ap.parse_args()

    with open(args.src) as f:
        src = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    train = convert(src, "train", args.legible_only, args.keep_empty)
    val = convert(src, "val", args.legible_only, args.keep_empty)
    train_dev, dev = split_coco(train, args.dev_images, args.seed)
    outputs = {
        "cocotext_train.json": train, "cocotext_train_dev.json": train_dev,
        "cocotext_dev.json": dev, "cocotext_val.json": val,
    }
    for name, d in outputs.items():
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(d, f)
        print(f"{name}: {len(d['images'])} images, {len(d['annotations'])} boxes")


if __name__ == "__main__":
    main()
