#!/usr/bin/env python3
# Build key-list files for train_textnet.py (CPU only; source LMDBs are opened
# read-only with lock=False and never modified).
#
# Each output file data_cache/<name>.keys.json holds
#   {"sources": {"<lmdb path>": [key, ...], ...}, ...stats}
# listing the samples whose label passes the training filter (alphabet 0-9a-z after
# lower-casing, 1..max_len characters). Synthetic sets are reduced to a fixed random
# subset (seeded). train_textnet.py takes these files directly as --train sources.
#
#   python tools/prep_textnet_data.py --out data_cache

import argparse
import json
import os
import time

import lmdb
import numpy as np

DATA = "/home/deepak/Documents/qacspnet-TRAIN/ext_data"
ALPHABET = set("0123456789abcdefghijklmnopqrstuvwxyz")

# name -> (list of LMDB folders, subset size or None for all)
SOURCES = {
    "cocotext_v2": ([f"{DATA}/cocotext_v2"], None),
    "iiit5k_train": ([f"{DATA}/iiit5k_train"], None),
    "ic13_train": ([f"{DATA}/ic13_train"], None),
    "ic15_train": ([f"{DATA}/ic15_train"], None),
    "svt_train": ([f"{DATA}/svt_train"], None),
    "mlt19_latin": ([f"{DATA}/mlt19_train"], None),
    "mjsynth_200k": ([f"{DATA}/mjsynth/chunk_{i:02d}" for i in range(20)], 200_000),
    "synthtext_200k": ([f"{DATA}/synthtext/train_{i:02d}" for i in range(20)], 200_000),
}


def valid(label, max_len):
    w = label.strip().lower()
    return 0 < len(w) <= max_len and all(ch in ALPHABET for ch in w)


def scan(path, max_len):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    keys = []
    with env.begin() as txn:
        n = int(txn.get(b"num-samples"))
        for i in range(1, n + 1):
            lab = txn.get(f"label-{i:09d}".encode())
            if lab is not None and valid(lab.decode("utf-8", "ignore"), max_len):
                keys.append(i)
    env.close()
    return n, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_cache")
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="subset of source names")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, (paths, subset) in SOURCES.items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        per, total, kept = {}, 0, 0
        for p in paths:
            n, keys = scan(p, args.max_len)
            per[p], total, kept = keys, total + n, kept + len(keys)
        if subset is not None and kept > subset:
            # fixed random subset over all folders together
            flat = [(p, k) for p in paths for k in per[p]]
            pick = np.random.default_rng(args.seed).choice(len(flat), subset, replace=False)
            pick.sort()
            per = {p: [] for p in paths}
            for j in pick:
                p, k = flat[j]
                per[p].append(int(k))
        used = sum(len(v) for v in per.values())
        out = os.path.join(args.out, f"{name}.keys.json")
        with open(out + ".tmp", "w") as f:
            json.dump({"name": name, "total": total, "valid": kept, "used": used,
                       "seed": args.seed if subset else None, "max_len": args.max_len,
                       "sources": {p: v for p, v in per.items() if v}}, f)
        os.replace(out + ".tmp", out)
        print(f"{name:16s} total={total:9d} valid={kept:9d} used={used:7d} "
              f"({time.time() - t0:.0f}s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
