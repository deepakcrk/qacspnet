#!/usr/bin/env python3
# Words whose crop contains symbols / punctuation next to the letters (e.g. "hello,", "(2014)",
# "www.site.com"): keep the crop, strip the symbols from the label ("hello", "2014", "wwwsitecom"),
# as IIIT5K's labels are made. CPU only; source LMDBs are read-only. Output: data_cache/symwords.keys.json
# with per-key label overrides.
import json, os, re, sys, time
import lmdb, numpy as np
DATA = "/home/deepak/Documents/qacspnet-TRAIN/ext_data"
SRC = {"cocotext_v2": ([f"{DATA}/cocotext_v2"], None), "mlt19_latin": ([f"{DATA}/mlt19_train"], None),
       "synthtext": ([f"{DATA}/synthtext/train_{i:02d}" for i in range(20)], 150_000)}
ALNUM = re.compile(r"[0-9a-z]")
def convert(label):
    w = label.strip().lower()
    if not w or all(ch.isalnum() for ch in w) or any(ord(ch) > 127 for ch in w):
        return None
    core = "".join(ch for ch in w if ALNUM.match(ch))
    if not (1 <= len(core) <= 25) or len(core) < 0.5 * len(w.replace(" ", "")):
        return None  # mostly symbols: skip
    return core
def main():
    out_src, out_lab, stats = {}, {}, {}
    for name, (paths, limit) in SRC.items():
        found = []
        t0 = time.time()
        for p in paths:
            env = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as t:
                n = int(t.get(b"num-samples"))
                for i in range(1, n + 1):
                    lab = t.get(f"label-{i:09d}".encode())
                    if lab is None: continue
                    c = convert(lab.decode("utf-8", "ignore"))
                    if c: found.append((p, i, c))
            env.close()
        if limit and len(found) > limit:
            idx = np.random.default_rng(0).choice(len(found), limit, replace=False); found = [found[j] for j in sorted(idx)]
        for p, i, c in found:
            out_src.setdefault(p, []).append(i); out_lab.setdefault(p, {})[str(i)] = c
        stats[name] = len(found); print(f"{name}: {len(found)} symbol-words ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"name": "symwords", "used": sum(stats.values()), "per_source": stats, "max_len": 25,
               "note": "labels = alnum core of the original label", "sources": out_src, "labels": out_lab},
              open("data_cache/symwords.keys.json", "w"))
    print("total", sum(stats.values()), "-> data_cache/symwords.keys.json")
if __name__ == "__main__": main()
