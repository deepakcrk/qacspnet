#!/usr/bin/env python3
# Flag likely-mislabelled training images: the model's prediction differs strongly from
# the label while the model is very confident. Uses training data only (no test data).
# CPU only. Writes data_cache/<name>_clean.keys.json (+ a removed-samples list).
#   python tools/clean_labels.py --run outputs/s4b_lowlr --sets cocotext_v2 synthtext_200k mlt19_latin
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_textnet as T
from qacspnet.models import QACSPTextNet


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.95, help="min mean char confidence")
    ap.add_argument("--min-ned", type=float, default=0.5, help="min normalized edit distance")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = json.load(open(os.path.join(a.run, "model_config.json")))
    m = QACSPTextNet(**cfg)
    m.load_state_dict(torch.load(os.path.join(a.run, "best.pth"), map_location="cpu"))
    m.eval()
    for name in a.sets:
        spec = json.load(open(f"data_cache/{name}.keys.json"))
        keep, removed, t0 = {}, [], time.time()
        for path, keys in spec["sources"].items():
            ds = T.LmdbDataset(path, img_size=m.img_size, alphabet=m.alphabet, max_len=25, keys=keys)
            ds.keep_ratio = True
            dl = torch.utils.data.DataLoader(ds, 256, collate_fn=T.collate, num_workers=2)
            idx = 0
            with torch.no_grad():
                for imgs, texts in dl:
                    lp = m(imgs).float().softmax(-1)  # T x B x C
                    conf, best = lp.max(-1)
                    preds = m.decode(lp.log())
                    for b, (p, t) in enumerate(zip(preds, texts)):
                        k = keys[idx]; idx += 1
                        nonblank = best[:, b] != 0
                        c = float(conf[nonblank, b].mean()) if nonblank.any() else 0.0
                        ned = edit(p, t) / max(len(t), 1)
                        if p != t and c >= a.conf and ned >= a.min_ned:
                            removed.append([path, k, t, p, round(c, 3)])
                        else:
                            keep.setdefault(path, []).append(k)
            print(f"{name} {os.path.basename(path)}: {idx} checked, removed so far {len(removed)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        used = sum(len(v) for v in keep.values())
        out = f"data_cache/{name}_clean.keys.json"
        json.dump({**{k: v for k, v in spec.items() if k != "sources"}, "name": f"{name}_clean",
                   "used": used, "cleaned_by": a.run, "removed": len(removed), "sources": keep},
                  open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        json.dump(removed, open(f"data_cache/{name}_removed.json", "w"))
        print(f"== {name}: kept {used}, removed {len(removed)} ({100*len(removed)/max(1,used+len(removed)):.1f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
