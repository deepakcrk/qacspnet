#!/usr/bin/env python3
# Background pools for the renderer, avoiding real text (CPU; sources are read-only).
#   ANPR detection images (number plates from the YOLO labels) + COCO 2014 train images
#   (all COCO-Text v2 regions, any legibility). Each pooled image is stored downscaled
#   (longest side 320, JPEG) with its text boxes in pixel coordinates, so crops can avoid them.
#   Output: data_cache/bg_full.pkl  [(jpeg_bytes, [[x0,y0,x1,y1], ...]), ...]
#           data_cache/bg_patches.pkl  tiny strips (4-40 px) that do not touch any text box
import json, os, pickle, random
from multiprocessing import Pool
import cv2, numpy as np
ANPR = "/home/deepak/ANPR/dataset_master/detection_master"
COCO = "/home/deepak/ANPR/dataset/coco2014"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")

def anpr_boxes(name, w, h):
    lab = os.path.join(ANPR, "labels", "train", os.path.splitext(name)[0] + ".txt")
    boxes = []
    if os.path.isfile(lab):
        for line in open(lab):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = map(float, p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes

def work(job):
    path, boxes = job
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    if boxes is None:
        boxes = anpr_boxes(os.path.basename(path), w, h)
    s = 320 / max(h, w)
    im = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    boxes = [[b[0] * s - 3, b[1] * s - 3, b[2] * s + 3, b[3] * s + 3] for b in boxes]  # 3 px margin
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    # tiny strips that avoid every box
    rng = np.random.default_rng(abs(hash(path)) % (2 ** 32)); H, W = im.shape[:2]; strips = []
    for _ in range(6):
        ph, pw = int(rng.integers(4, 41)), int(rng.integers(4, 41))
        y, x = int(rng.integers(0, max(1, H - ph))), int(rng.integers(0, max(1, W - pw)))
        if any(not (x + pw <= b[0] or x >= b[2] or y + ph <= b[1] or y >= b[3]) for b in boxes):
            continue
        p = im[y:y + ph, x:x + pw]
        if p.size and p.std() > 2:
            strips.append(np.ascontiguousarray(p))
        if len(strips) >= 2:
            break
    return buf.tobytes(), boxes, strips

def main():
    rnd = random.Random(7)
    anpr_dir = os.path.join(ANPR, "images", "train")
    jobs = [(os.path.join(anpr_dir, f), None) for f in rnd.sample(sorted(os.listdir(anpr_dir)), 3000)]
    ct = json.load(open(os.path.join(COCO, "cocotext.v2.json")))
    by_img = {}
    for a in ct["anns"].values():
        by_img.setdefault(str(a["image_id"]), []).append(a["bbox"])
    ids = sorted(k for k, v in ct["imgs"].items() if v.get("set") in ("train", "val"))
    for iid in rnd.sample(ids, 3000):
        info = ct["imgs"][iid]
        boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh in by_img.get(iid, [])]
        jobs.append((os.path.join(COCO, "images", "train2014", info["file_name"]), boxes))
    with Pool(8) as pool:
        res = [r for r in pool.map(work, jobs, chunksize=40) if r]
    full = [(b, bx) for b, bx, _ in res]
    strips = [s for _, _, ss in res for s in ss]
    pickle.dump(full, open(os.path.join(OUT, "bg_full.pkl"), "wb"))
    pickle.dump(strips, open(os.path.join(OUT, "bg_patches.pkl"), "wb"))
    nbox = sum(len(bx) > 0 for _, bx in full)
    print(f"full images: {len(full)} ({nbox} with text boxes to avoid); strips: {len(strips)}")

if __name__ == "__main__":
    main()
