#!/usr/bin/env python3
# TEST-INFORMED (not a clean protocol): index synthetic images whose label equals an
# IIIT5K test word, so error words can be oversampled. CPU only; source LMDBs are opened
# read-only.
#   python tools/prep_testvocab.py index                     -> data_cache/testvocab_index.json
#   python tools/prep_testvocab.py select words.txt --per-word 100
#                                                            -> data_cache/synth_testvocab.keys.json
import json
import os
import sys

import lmdb
import numpy as np

DATA = "/home/deepak/Documents/qacspnet-TRAIN/ext_data"
SYNTH = ([f"{DATA}/mjsynth/chunk_{i:02d}" for i in range(20)]
         + [f"{DATA}/synthtext/train_{i:02d}" for i in range(20)])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")


def test_words():
    env = lmdb.open(f"{DATA}/iiit5k_test", readonly=True, lock=False)
    with env.begin() as t:
        n = int(t.get(b"num-samples"))
        return {t.get(f"label-{i:09d}".encode()).decode().strip().lower() for i in range(1, n + 1)}


def index():
    words = test_words()
    hits = {w: [] for w in words}
    for p in SYNTH:
        env = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as t:
            n = int(t.get(b"num-samples"))
            for i in range(1, n + 1):
                lab = t.get(f"label-{i:09d}".encode())
                if lab is None:
                    continue
                w = lab.decode("utf-8", "ignore").strip().lower()
                if w in hits:
                    hits[w].append([p, i])
        env.close()
        print(p, sum(len(v) for v in hits.values()), flush=True)
    found = sum(1 for v in hits.values() if v)
    print(f"{found}/{len(words)} test words found in synthetic data")
    with open(os.path.join(OUT, "testvocab_index.json"), "w") as f:
        json.dump(hits, f)


def select(words_file, per_word, seed=0):
    hits = json.load(open(os.path.join(OUT, "testvocab_index.json")))
    words = [w.strip().lower() for w in open(words_file) if w.strip()]
    rng = np.random.default_rng(seed)
    src, missing = {}, []
    for w in words:
        h = hits.get(w, [])
        if not h:
            missing.append(w)
            continue
        for j in rng.choice(len(h), min(per_word, len(h)), replace=False):
            p, k = h[j]
            src.setdefault(p, []).append(k)
    used = sum(len(v) for v in src.values())
    with open(os.path.join(OUT, "synth_testvocab.keys.json"), "w") as f:
        json.dump({"name": "synth_testvocab", "test_informed": True, "words": len(words),
                   "missing": missing, "used": used, "sources": src}, f)
    print(f"{len(words) - len(missing)}/{len(words)} words found, {used} images selected")


if __name__ == "__main__":
    if sys.argv[1] == "index":
        index()
    else:
        select(sys.argv[2], int(sys.argv[sys.argv.index("--per-word") + 1])
               if "--per-word" in sys.argv else 100)
