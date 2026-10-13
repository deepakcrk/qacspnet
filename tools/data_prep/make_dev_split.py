#!/usr/bin/env python3
"""Split a COCO-format training annotation file into train / dev by image.

The dev split is used for model selection (checkpoints, hyper-parameters) and
the official validation / test split is used only once for the final numbers.

  python tools/data_prep/make_dev_split.py \
      --ann datasets/COCO/annotations/instances_train2017.json \
      --dev-images 5000 --seed 0 \
      --out-train datasets/COCO/annotations/instances_train2017_minusdev.json \
      --out-dev datasets/COCO/annotations/instances_dev2017.json
"""
import argparse
import json
import random


def split_coco(ann, n_dev, seed=0):
    ids = sorted(img["id"] for img in ann["images"])
    random.Random(seed).shuffle(ids)
    dev_ids = set(ids[:n_dev])

    def subset(keep):
        return {
            **{k: v for k, v in ann.items() if k not in ("images", "annotations")},
            "images": [im for im in ann["images"] if keep(im["id"])],
            "annotations": [a for a in ann["annotations"] if keep(a["image_id"])],
        }

    return subset(lambda i: i not in dev_ids), subset(lambda i: i in dev_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--dev-images", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    args = ap.parse_args()

    with open(args.ann) as f:
        ann = json.load(f)
    train, dev = split_coco(ann, args.dev_images, args.seed)
    for path, d in ((args.out_train, train), (args.out_dev, dev)):
        with open(path, "w") as f:
            json.dump(d, f)
        print(f"{path}: {len(d['images'])} images, {len(d['annotations'])} boxes")


if __name__ == "__main__":
    main()
