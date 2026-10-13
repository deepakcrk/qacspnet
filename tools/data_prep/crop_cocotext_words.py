#!/usr/bin/env python3
"""Crop word images from COCO-Text v2 for QA-CSPTextNet training / model selection.

Kept: legible, English, transcription made only of characters in --alphabet
(default 0-9 a-z; lower-cased unless the alphabet has upper-case letters),
1 to --max-len characters, box height >= --min-h.
The image split follows the detection split from convert_cocotext.py, so no image
is shared between train / dev / val:

  <out>/train/*.jpg + train.txt   images of cocotext_train_dev.json
  <out>/dev/*.jpg   + dev.txt     images of cocotext_dev.json  (model selection)
  <out>/val/*.jpg   + val.txt     images of cocotext_val.json  (official val)

Label files: "<split>/<name>.jpg <word>" per line (tools/train_textnet.py format).

  python tools/data_prep/crop_cocotext_words.py \
      --src path/to/cocotext.v2.json \
      --images path/to/coco2014/train2014 \
      --ann-dir datasets/COCOText/annotations --out datasets/COCOText_words
"""
import argparse
import json
import os

import cv2

DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
SPLITS = {"train": "cocotext_train_dev.json", "dev": "cocotext_dev.json",
          "val": "cocotext_val.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cocotext.v2.json (has transcriptions)")
    ap.add_argument("--images", required=True, help="COCO train2014 image folder")
    ap.add_argument("--ann-dir", required=True, help="output folder of convert_cocotext.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.05, help="padding, fraction of box height")
    ap.add_argument("--min-h", type=int, default=8)
    ap.add_argument("--alphabet", default=DEFAULT_ALPHABET,
                    help="characters to keep: the characters themselves or a text file "
                         "holding them; labels keep their case if it has upper-case letters")
    ap.add_argument("--max-len", type=int, default=25)
    args = ap.parse_args()
    alphabet = args.alphabet
    if os.path.isfile(alphabet):
        with open(alphabet, encoding="utf-8") as f:
            alphabet = "".join(line.rstrip("\n") for line in f)
    charset = set(alphabet)
    case_sensitive = any(ch != ch.lower() for ch in alphabet)

    with open(args.src) as f:
        src = json.load(f)

    for split, ann_name in SPLITS.items():
        with open(os.path.join(args.ann_dir, ann_name)) as f:
            img_ids = {im["id"] for im in json.load(f)["images"]}
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        lines, skipped = [], {"illegible": 0, "language": 0, "charset": 0, "size": 0}
        for img_id in sorted(img_ids):
            info = src["imgs"][str(img_id)]
            img = None
            for aid in src["imgToAnns"].get(str(img_id), []):
                a = src["anns"][str(aid)]
                word = a.get("utf8_string", "").strip()
                if not case_sensitive:
                    word = word.lower()
                if a.get("legibility") != "legible":
                    skipped["illegible"] += 1
                    continue
                if a.get("language") != "english":
                    skipped["language"] += 1
                    continue
                if not (1 <= len(word) <= args.max_len and all(ch in charset for ch in word)):
                    skipped["charset"] += 1
                    continue
                x, y, w, h = a["bbox"]
                if h < args.min_h or w < 2:
                    skipped["size"] += 1
                    continue
                if img is None:
                    img = cv2.imread(os.path.join(args.images, info["file_name"]))
                p = args.pad * h
                x1, y1 = max(int(x - p), 0), max(int(y - p), 0)
                x2 = min(int(x + w + p + 0.5), img.shape[1])
                y2 = min(int(y + h + p + 0.5), img.shape[0])
                name = f"{split}/{img_id}_{aid}.jpg"
                cv2.imwrite(os.path.join(args.out, name), img[y1:y2, x1:x2],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                lines.append(f"{name} {word}")
        with open(os.path.join(args.out, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{split}: {len(lines)} words from {len(img_ids)} images, skipped {skipped}")


if __name__ == "__main__":
    main()
