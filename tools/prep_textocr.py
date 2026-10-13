#!/usr/bin/env python3
# TextOCR (train json) -> word-crop LMDB in data_cache/textocr_train (CPU only, read-only source).
# Keeps legible words whose lower-cased label is 0-9a-z, 1..25 chars. Crops by the annotated
# quadrilateral (perspective warp) when it has 4 points, else by the axis-aligned bbox.
#   python tools/prep_textocr.py --src /home/deepak/ANPR/dataset/textOCR --out data_cache/textocr_train
import argparse
import json
import os
import re
import time
from multiprocessing import Pool

import cv2
import lmdb
import numpy as np

OK = re.compile(r"[0-9a-z]{1,25}")


def crop_quad(img, pts):
    q = np.asarray(pts, np.float32).reshape(4, 2)
    sm, df = q.sum(1), q[:, 1] - q[:, 0]
    q = q[[sm.argmin(), df.argmin(), sm.argmax(), df.argmax()]]  # TL, TR, BR, BL
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    if w < 4 or h < 4:
        return None
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def work(job):
    path, anns = job
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    out = []
    if img is None:
        return out
    for word, pts, bbox in anns:
        c = crop_quad(img, pts) if len(pts) == 8 else None
        if c is None:
            x, y, w, h = bbox
            if w < 4 or h < 4:
                continue
            c = img[max(0, int(y)):int(y + h) + 1, max(0, int(x)):int(x + w) + 1]
        if c is None or c.size == 0 or c.shape[0] < 4 or c.shape[1] < 4:
            continue
        if c.shape[0] > 128:  # keep crops small (height 128 is plenty for 64-px inputs)
            s = 128 / c.shape[0]
            c = cv2.resize(c, (max(4, int(c.shape[1] * s)), 128), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            out.append((buf.tobytes(), word))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symbol-words", action="store_true",
                    help="instead: words with symbols next to the letters; the label is the "
                         "alnum core (symbols stripped), as IIIT5K labels are made")
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.src, "TextOCR_0.1_train.json")))
    jobs = {}
    for ann in d["anns"].values():
        w = ann["utf8_string"].strip().lower()
        if ann["utf8_string"] == ".":
            continue
        if a.symbol_words:
            if all(ch.isalnum() for ch in w) or any(ord(ch) > 127 for ch in w):
                continue
            core = "".join(ch for ch in w if OK.fullmatch(ch))
            if not (1 <= len(core) <= 25) or len(core) < 0.5 * len(w.replace(" ", "")):
                continue
            w = core
        elif not OK.fullmatch(w):
            continue
        img = d["imgs"][ann["image_id"]]
        path = os.path.join(a.src, "train_val_images", "train_images", os.path.basename(img["file_name"]))
        jobs.setdefault(path, []).append((w, ann["points"], ann["bbox"]))
    jobs = sorted(jobs.items())
    print(f"{len(jobs)} images, {sum(len(v) for _, v in jobs)} words", flush=True)
    os.makedirs(a.out, exist_ok=True)
    env = lmdb.open(a.out, map_size=int(12e9))
    n, t0 = 0, time.time()
    with Pool(a.workers) as pool:
        txn = env.begin(write=True)
        for i, crops in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            for buf, word in crops:
                n += 1
                txn.put(f"image-{n:09d}".encode(), buf)
                txn.put(f"label-{n:09d}".encode(), word.encode())
            if i % 500 == 0:
                txn.commit(); txn = env.begin(write=True)
                print(f"{i}/{len(jobs)} images, {n} crops, {time.time() - t0:.0f}s", flush=True)
        txn.put(b"num-samples", str(n).encode())
        txn.commit()
    env.sync(); env.close()
    stem = os.path.basename(a.out.rstrip("/"))
    keys = os.path.join(os.path.dirname(a.out.rstrip("/")), f"{stem}.keys.json")
    with open(keys, "w") as f:
        json.dump({"name": stem, "total": n, "valid": n, "used": n, "seed": None,
                   "max_len": 25, "sources": {os.path.abspath(a.out): list(range(1, n + 1))}}, f)
    print(f"done: {n} crops -> {a.out}; keys {keys} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
