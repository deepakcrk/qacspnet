#!/usr/bin/env python3
# COCO 2014 images (downscaled, with COCO-Text boxes) for the wiggly real-photo pieces used as
# decorations / objects next to text. Output: data_cache/bg_coco.pkl [(jpeg, boxes), ...]
import json, os, pickle, random
from multiprocessing import Pool
import cv2
COCO = "/home/deepak/ANPR/dataset/coco2014"
def work(job):
    path, boxes = job
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]; s = 320 / max(h, w)
    im = cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (buf.tobytes(), [[b[0] * s - 3, b[1] * s - 3, b[2] * s + 3, b[3] * s + 3] for b in boxes]) if ok else None
def main():
    ct = json.load(open(os.path.join(COCO, "cocotext.v2.json")))
    by = {}
    for a in ct["anns"].values():
        x, y, bw, bh = a["bbox"]; by.setdefault(str(a["image_id"]), []).append([x, y, x + bw, y + bh])
    ids = random.Random(11).sample(sorted(ct["imgs"]), 4000)
    jobs = [(os.path.join(COCO, "images", "train2014", ct["imgs"][i]["file_name"]), by.get(i, [])) for i in ids]
    with Pool(8) as p:
        res = [r for r in p.map(work, jobs, chunksize=50) if r]
    pickle.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache", "bg_coco.pkl"), "wb"))
    print(len(res), "COCO images")
if __name__ == "__main__":
    main()
