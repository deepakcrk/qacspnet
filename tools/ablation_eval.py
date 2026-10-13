"""Evaluation-only analyses of one trained QA-CSPTextNet run (no training). User 2026-10-11.
   python tools/ablation_eval.py <run_dir> <out_prefix> [--limit N] [--only E1,E2,...]
Writes <out_prefix>.md, <out_prefix>.json, <out_prefix>_E7.png.

E1 degradation robustness: Gaussian blur, motion blur, noise, JPEG, downscale, low contrast,
   5 severities each, on COCO-Text v2 val and IIIT5K test (applied to the raw crop before resizing).
E2 accuracy by image quality: quintiles of sharpness (variance of Laplacian), crop height, contrast.
E3 cross-dataset: IC13 (857), IC15 (2077), SVT (647) test, plus COCO val and IIIT5K (clean).
E4 QGE knockout at test time: alpha = 0 in all / backbone / neck / each single gate.
E5 QGE quality-gate vs image quality: Spearman correlation per gate on COCO val.
E6 fusion-level knockout: zero P8 / P16 / both inside the fusion block, or the whole fusion branch.
E7 figure: input, QGE spatial attention and gated map on easy and hard COCO val images.
All evaluations: FP32, greedy CTC, same preprocessing as training evaluation (keep-ratio from preprocess.json)."""
import sys, os, json, time, argparse, collections
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import train_textnet as T
from qacspnet.models import QACSPTextNet
from qacspnet.models import qacsp_textnet as QT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "ext_data")
SETS = {"coco_val": os.path.join(ROOT, "data_cache/cocotext_val.keys.json"), "iiit5k": os.path.join(EXT, "iiit5k_test")}
CROSS = {"ic13_857": os.path.join(EXT, "ic13_test_857"), "ic15_2077": os.path.join(EXT, "ic15_test_2077"),
         "svt": os.path.join(EXT, "svt_test")}


# ------------------------------------------------------------------ degradations (gray uint8 in/out)
def _motion(img, k):
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, ker)

def _noise(img, s, seed):
    return np.clip(img.astype(np.float32) + np.random.default_rng(seed).normal(0, s, img.shape), 0, 255).astype(np.uint8)

def _jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)

def _down(img, f):
    h, w = img.shape[:2]
    s = cv2.resize(img, (max(2, round(w * f)), max(2, round(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

def _contrast(img, c):
    m = img.mean(); return np.clip(m + (img.astype(np.float32) - m) * c, 0, 255).astype(np.uint8)

DEGS = {
    "gauss_blur (sigma)": [0.5, 1.0, 1.5, 2.0, 3.0],
    "motion_blur (kernel)": [3, 5, 7, 9, 13],
    "noise (std)": [5, 10, 20, 30, 45],
    "jpeg (quality)": [50, 30, 20, 10, 5],
    "downscale (factor)": [0.75, 0.5, 0.35, 0.25, 0.15],
    "contrast (factor)": [0.7, 0.5, 0.35, 0.25, 0.15],
}


def apply_deg(kind, v, im, j):
    if kind.startswith("gauss"): return cv2.GaussianBlur(im, (0, 0), v)
    if kind.startswith("motion"): return _motion(im, v)
    if kind.startswith("noise"): return _noise(im, v, j)
    if kind.startswith("jpeg"): return _jpeg(im, v)
    if kind.startswith("down"): return _down(im, v)
    return _contrast(im, v)


class Wrap(torch.utils.data.Dataset):
    """Raw crop -> optional degradation -> the training preprocessing (prepare) ; label as in evaluation."""
    def __init__(self, base, deg=None, limit=None):
        self.parts = list(getattr(base, "datasets", [base]))
        self.idx = [(p, i) for p in self.parts for i in range(len(p))]
        if limit: self.idx = self.idx[:limit]
        self.deg = deg

    def __len__(self): return len(self.idx)

    def __getitem__(self, j):
        p, i = self.idx[j]
        img = p.raw(i)
        if self.deg is not None:  # (kind, value), picklable for DataLoader workers
            img = apply_deg(self.deg[0], self.deg[1], img, j)
        k = p.index[i]
        if p.env is None: p.env = T._open_lmdb(p.path)
        lab = p.labels.get(k) if p.labels else None
        if lab is None:
            with p.env.begin() as txn:
                lab = txn.get(f"label-{k:09d}".encode()).decode("utf-8", "ignore")
        return T.prepare(p, img, i), T._clean(lab, p.alphabet, p.max_len)


class Ctx:
    def __init__(self, run, limit, keep_ratio):
        self.run, self.limit, self.kr = run, limit, keep_ratio
        cfg = json.load(open(f"{run}/model_config.json")); self.cfg = cfg
        self.m = QACSPTextNet(**cfg); self.m.load_state_dict(torch.load(f"{run}/best.pth", map_location="cpu"))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.m.to(self.dev).eval()
        self.base = {}

    def ds(self, path):
        if path not in self.base:
            d = T.build_dataset(os.path.join(ROOT, "data_cache"), path, None, 25, self.m.img_size)
            for p in getattr(d, "datasets", [d]):
                p.keep_ratio = self.kr; p.color = self.cfg.get("in_chans", 1) == 3
            self.base[path] = d
        return self.base[path]

    def preds(self, path, deg=None):
        """-> list of (pred, gt) in dataset order."""
        for p in getattr(self.ds(path), "datasets", [self.ds(path)]):
            p.env = None  # LMDB handles opened in this process cannot be pickled to DataLoader workers
        dl = torch.utils.data.DataLoader(Wrap(self.ds(path), deg, self.limit), 256, num_workers=8, collate_fn=T.collate)
        out = []
        with torch.inference_mode():
            for imgs, texts in dl:
                for p, t in zip(self.m.recognize(imgs.to(self.dev)), texts):
                    out.append((p, "".join(c for c in self.m.normalize(t) if c in self.m.alphabet)))
        return out

    def acc(self, path, deg=None):
        r = self.preds(path, deg); return 100.0 * sum(p == g for p, g in r) / max(1, len(r))


def qge_modules(m):
    return [(n.replace("backbone.backbone.", "bb.").replace("backbone.", "neck.").replace(".qge", ""), mod)
            for n, mod in m.named_modules() if n.endswith(".qge")]


def set_alpha(mods, value_by_name):
    """Temporarily set alpha (free/fixed: value; bounded: logit). Returns restore list."""
    saved = []
    for n, q in mods:
        if n in value_by_name:
            saved.append((q, q.alpha.data.clone()))
            q.alpha.data.fill_(value_by_name[n])
    return saved


def restore(saved):
    for q, a in saved: q.alpha.data.copy_(a)


# ------------------------------------------------------------------ analyses
def E1(c):
    res = {}
    for name, path in SETS.items():
        clean = c.acc(path); res[name] = {"clean": clean}
        for dn, vals in DEGS.items():
            res[name][dn] = [(v, c.acc(path, (dn, v))) for v in vals]
            print(f"  E1 {name} {dn}: {[round(a, 2) for _, a in res[name][dn]]}", flush=True)
    return res


def quality(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()), "height": float(g.shape[0]), "contrast": float(g.std())}


def per_image_quality(c, path):
    w = Wrap(c.ds(path), None, c.limit)
    return [quality(p.raw(i)) for p, i in w.idx]


def E2(c, cache):
    res = {}
    for name, path in SETS.items():
        r = c.preds(path); ok = np.array([p == g for p, g in r]); q = cache.setdefault(name, per_image_quality(c, path))
        res[name] = {}
        for key in ("sharpness", "height", "contrast"):
            v = np.array([x[key] for x in q]); edges = np.quantile(v, [0, .2, .4, .6, .8, 1])
            bins = []
            for b in range(5):
                sel = (v >= edges[b]) & ((v <= edges[b + 1]) if b == 4 else (v < edges[b + 1]))
                bins.append({"range": [float(edges[b]), float(edges[b + 1])], "n": int(sel.sum()),
                             "acc": float(100 * ok[sel].mean()) if sel.any() else None})
            res[name][key] = bins
        print(f"  E2 {name} done", flush=True)
    return res


def E3(c):
    res = {k: c.acc(p) for k, p in {**SETS, **CROSS}.items()}
    print(f"  E3 {res}", flush=True); return res


def E4(c):
    mods = qge_modules(c.m); names = [n for n, _ in mods]
    groups = {"none (as trained)": [], "all gates": names,
              "backbone gates": [n for n in names if n.startswith("bb.")], "neck gates": [n for n in names if n.startswith("neck.")]}
    groups.update({f"only {n}": [n] for n in names})
    alpha_learned = {n: float(q.blend_factor()) for n, q in mods}
    res = {"alpha": alpha_learned, "knockout": {}}
    for g, sel in groups.items():
        saved = set_alpha(mods, {n: 0.0 for n in sel}) if any(q.alpha_mode != "bounded" for _, q in mods) else []
        if any(q.alpha_mode == "bounded" for _, q in mods):
            saved = set_alpha(mods, {n: -30.0 for n in sel})  # sigmoid(-30) ~ 0
        res["knockout"][g] = {k: c.acc(p) for k, p in SETS.items()}
        restore(saved)
        print(f"  E4 {g}: {res['knockout'][g]}", flush=True)
    return res


def gate_values(c, path, n_max=None, deg=None):
    """Mean quality-gate value (1-w or w) per image and per QGE gate (global type)."""
    mods = qge_modules(c.m); vals = collections.defaultdict(list); hooks = []
    for n, q in mods:
        def pre(mod, inp, n=n):
            if not mod.use_quality: return
            x = inp[0]; e = mod.edge(x).abs()
            z = e.mean(dim=(2, 3)) if mod.qge_type == "global" else e.permute(0, 2, 3, 1)
            w = torch.sigmoid(mod.fc2(torch.nn.functional.silu(mod.fc1(z))))
            a = 1.0 - w if mod.invert else w
            vals[n].extend(a.reshape(a.shape[0], -1).mean(1).float().cpu().tolist())
        hooks.append(q.register_forward_pre_hook(pre))
    lim = c.limit
    if n_max: c.limit = n_max if not lim else min(lim, n_max)
    try:
        c.preds(path, deg)
    finally:
        c.limit = lim
    for h in hooks: h.remove()
    return vals


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def E5(c, cache):
    path = SETS["coco_val"]; vals = gate_values(c, path); q = cache.setdefault("coco_val", per_image_quality(c, path))
    res = {}
    for n, v in vals.items():
        v = np.array(v[:len(q)])
        res[n] = {k: spearman(v, np.array([x[k] for x in q[:len(v)]])) for k in ("sharpness", "height", "contrast")}
    print(f"  E5 done ({len(res)} gates)", flush=True)
    return res


def E6(c):
    m = c.m
    if getattr(m, "level_fusion", None) != "fpn":
        return {"skipped": "level_fusion is not fpn"}
    mode = {"keep": [1, 1, 1], "branch": 1.0}
    orig = m.encode_features

    def enc(x):
        feats = m.backbone(x); f0 = feats[0]; h0, w0 = f0.shape[-2:]; maps = [f0]
        for lat, f in zip(m.fuse_lat, feats[1:]):
            f = lat(f); f = f @ QT._linear_upsample_matrix(f.shape[-1], w0, f.device, f.dtype)
            f = (f.transpose(-1, -2) @ QT._linear_upsample_matrix(f.shape[-2], h0, f.device, f.dtype)).transpose(-1, -2)
            maps.append(f)
        maps = [mp * k for mp, k in zip(maps, mode["keep"])]
        f = f0 + mode["branch"] * m.fuse_out(torch.cat(maps, 1))
        return m.rnn(m.fc_in(m._collapse(f).permute(2, 0, 1)))[0]

    if m.fpn_upsample != "bilinear":
        return {"skipped": "fpn_upsample is not bilinear"}
    m.encode_features = enc; res = {}
    for g, keep, br in [("full (as trained)", [1, 1, 1], 1.0), ("P8 zeroed", [1, 0, 1], 1.0), ("P16 zeroed", [1, 1, 0], 1.0),
                        ("P8 + P16 zeroed", [1, 0, 0], 1.0), ("fusion branch off (P4 only)", [1, 1, 1], 0.0)]:
        mode.update(keep=keep, branch=br); res[g] = {k: c.acc(p) for k, p in SETS.items()}
        print(f"  E6 {g}: {res[g]}", flush=True)
    m.encode_features = orig
    return res


def E8(c):
    """Per-stride evaluation: keep only some of P4 / P8 / P16. A level that is off is zeroed everywhere,
    including P4's residual (as the training-time level dropout does)."""
    m = c.m
    if getattr(m, "level_fusion", None) != "fpn" or m.fpn_upsample != "bilinear":
        return {"skipped": "needs fpn fusion with bilinear upsampling"}
    keep = [1, 1, 1]; orig = m.encode_features

    def enc(x):
        feats = m.backbone(x); f0 = feats[0]; h0, w0 = f0.shape[-2:]; maps = [f0]
        for lat, f in zip(m.fuse_lat, feats[1:]):
            f = lat(f); f = f @ QT._linear_upsample_matrix(f.shape[-1], w0, f.device, f.dtype)
            f = (f.transpose(-1, -2) @ QT._linear_upsample_matrix(f.shape[-2], h0, f.device, f.dtype)).transpose(-1, -2)
            maps.append(f)
        maps = [mp * k for mp, k in zip(maps, keep)]
        f = maps[0] + m.fuse_out(torch.cat(maps, 1))
        return m.rnn(m.fc_in(m._collapse(f).permute(2, 0, 1)))[0]

    m.encode_features = enc; res = {}
    for g, k in [("full (P4 + P8 + P16)", [1, 1, 1]), ("stride 4 only", [1, 0, 0]), ("stride 8 only", [0, 1, 0]),
                 ("stride 16 only", [0, 0, 1]), ("without stride 4 (P8 + P16)", [0, 1, 1]),
                 ("without stride 8 (P4 + P16)", [1, 0, 1]), ("without stride 16 (P4 + P8)", [1, 1, 0])]:
        keep[:] = k; res[g] = {s_: c.acc(p) for s_, p in SETS.items()}
        print(f"  E8 {g}: {res[g]}", flush=True)
    m.encode_features = orig
    return res


def E5S(c, n=2000):
    """Quality-gate response (mean over images, per gate and averaged) against corruption severity,
    on the first n COCO-Text val images (Reviewer 3: gate response versus corruption severity)."""
    path = SETS["coco_val"]; res = {}
    if not any(q.use_quality for _, q in qge_modules(c.m)):
        return {"skipped": "model has no quality branch"}
    clean = {g: float(np.mean(v)) for g, v in gate_values(c, path, n).items()}
    for dn, vals in DEGS.items():
        res[dn] = [{"level": 0, "mean": float(np.mean(list(clean.values()))), "per_gate": clean}]
        for v in vals:
            gv = {g: float(np.mean(x)) for g, x in gate_values(c, path, n, (dn, v)).items()}
            res[dn].append({"level": v, "mean": float(np.mean(list(gv.values()))), "per_gate": gv})
        print(f"  E5S {dn}: {[round(r['mean'], 4) for r in res[dn]]}", flush=True)
    return res


def E9(c, cache):
    """AFW: learned level weights (P4 / P8 / P16) per image, averaged per word-height quintile."""
    from qacspnet.models.qacsp_blocks import AdaptiveFPNWeighting
    afw = [mod for mod in c.m.modules() if isinstance(mod, AdaptiveFPNWeighting)]
    if not afw:
        return {"skipped": "no AFW"}
    res = {}
    for name, path in SETS.items():
        ws = []
        def hook(mod, inp, out):
            for b in range(out[0].shape[0]):
                ws.append([float((o[b] * i[b]).sum() / ((i[b] * i[b]).sum() + 1e-12)) for i, o in zip(inp[0], out)])
        h = afw[0].register_forward_hook(hook); c.preds(path); h.remove()
        q = cache.setdefault(name, per_image_quality(c, path)); ws = np.array(ws[:len(q)])
        hgt = np.array([x["height"] for x in q[:len(ws)]]); edges = np.quantile(hgt, [0, .2, .4, .6, .8, 1])
        bins = []
        for b in range(5):
            sel = (hgt >= edges[b]) & ((hgt <= edges[b + 1]) if b == 4 else (hgt < edges[b + 1]))
            bins.append({"height": [float(edges[b]), float(edges[b + 1])], "n": int(sel.sum()), "w": ws[sel].mean(0).round(4).tolist()})
        res[name] = {"overall": ws.mean(0).round(4).tolist(), "std": ws.std(0).round(4).tolist(), "by_height": bins}
        print(f"  E9 {name}: {res[name]['overall']}", flush=True)
    return res


def E7(c, out_png, cache):
    """Rows: 4 sharp correct + 4 blurry (correct or not) COCO val images. Columns: input | QGE spatial
    attention of the first and last backbone gate (upsampled) | gated input-gate map."""
    path = SETS["coco_val"]; r = c.preds(path); q = cache.setdefault("coco_val", per_image_quality(c, path))
    sh = np.array([x["sharpness"] for x in q]); ok = np.array([p == g for p, g in r]); order = np.argsort(sh)
    hard = [int(i) for i in order[:400][:4]]; easy = [int(i) for i in order[::-1] if ok[i]][:4]
    w = Wrap(c.ds(path), None, None); mods = dict(qge_modules(c.m))
    keys = [k for k in mods if k.startswith("bb.") and getattr(mods[k], "use_spatial", True)]
    if not keys:
        return {"skipped": "no spatial attention map in this model"}
    pick = [keys[0], keys[-1]] if keys else []
    maps = {}
    def hook(n):
        def f(mod, inp, out):
            x = inp[0]; fa = x.mean(1, keepdim=True); fm = x.amax(1, keepdim=True)
            maps[n] = torch.sigmoid(mod.spatial(torch.cat([fa, fm], 1)))[0, 0].float().cpu().numpy()
        return f
    hs = [mods[n].register_forward_hook(hook(n)) for n in pick]
    H, W = c.m.img_size; tiles = []
    for j in easy + hard:
        x, gt = w[j]
        with torch.inference_mode():
            pred = c.m.recognize(x[None].to(c.dev))[0]
        inp = ((x[0].numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        row = [cv2.cvtColor(inp, cv2.COLOR_GRAY2BGR)]
        for n in pick:
            a = cv2.resize(maps[n], (W, H)); a = (255 * (a - a.min()) / (np.ptp(a) + 1e-6)).astype(np.uint8)
            row.append(cv2.applyColorMap(a, cv2.COLORMAP_JET))
        row = np.hstack(row); bar = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{'easy' if j in easy else 'blurry'} sharp={sh[j]:.0f} GT={gt} pred={pred}", (4, 16), 0, 0.5,
                    (0, 120, 0) if pred == gt else (0, 0, 200), 1)
        tiles.append(np.vstack([bar, row]))
    for h in hs: h.remove()
    head = np.full((26, tiles[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"input | QGE spatial attention {' | '.join(pick)}   ({os.path.basename(c.run)})", (4, 18), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(out_png, np.vstack([head] + tiles))
    return {"figure": out_png, "easy_idx": easy, "blurry_idx": hard, "gates": pick}


# ------------------------------------------------------------------ report
def write_md(path, run, res, meta):
    L = [f"# Evaluation-only analyses: `{os.path.basename(run)}`", "", f"{meta['date']}; checkpoint `{run}/best.pth`; "
         f"input {meta['img_size']}, keep-ratio {meta['keep_ratio']}; FP32, greedy CTC; {meta['device']}.", ""]
    if "E3" in res:
        L += ["## E3 Clean accuracy and cross-dataset (word accuracy %)", "", "| " + " | ".join(res["E3"]) + " |",
              "|" + "---|" * len(res["E3"]), "| " + " | ".join(f"{v:.2f}" for v in res["E3"].values()) + " |", ""]
    if "E1" in res:
        L += ["## E1 Degradation robustness (word accuracy %, severity 1 -> 5)", ""]
        for s, d in res["E1"].items():
            L += [f"**{s}** (clean {d['clean']:.2f})", "", "| degradation | levels | s1 | s2 | s3 | s4 | s5 | mean drop |", "|---|---|---|---|---|---|---|---|"]
            for dn, pts in d.items():
                if dn == "clean": continue
                L.append(f"| {dn} | {', '.join(str(v) for v, _ in pts)} | " + " | ".join(f"{a:.2f}" for _, a in pts)
                         + f" | {d['clean'] - np.mean([a for _, a in pts]):.2f} |")
            L.append("")
    if "E2" in res:
        L += ["## E2 Accuracy by image-quality quintile (Q1 = lowest)", ""]
        for s, d in res["E2"].items():
            L += [f"**{s}**", "", "| measure | Q1 | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|"]
            for k, bins in d.items():
                L.append(f"| {k} | " + " | ".join(f"{b['acc']:.2f} ({b['range'][0]:.0f}-{b['range'][1]:.0f})" for b in bins) + " |")
            L.append("")
    if "E4" in res:
        L += ["## E4 QGE knockout at test time (alpha = 0)", "", "Learned alpha: " + ", ".join(f"{k} {v:.2f}" for k, v in res["E4"]["alpha"].items()), "",
              "| gates switched off | COCO val | IIIT5K |", "|---|---|---|"]
        for g, d in res["E4"]["knockout"].items():
            L.append(f"| {g} | {d['coco_val']:.2f} | {d['iiit5k']:.2f} |")
        L.append("")
    if "E5" in res:
        L += ["## E5 Quality gate vs image quality (Spearman rho, COCO val)", "", "| gate | sharpness | height | contrast |", "|---|---|---|---|"]
        for n, d in res["E5"].items():
            L.append(f"| {n} | {d['sharpness']:.3f} | {d['height']:.3f} | {d['contrast']:.3f} |")
        L.append("")
    if "E5S" in res and "skipped" not in res["E5S"]:
        L += ["## E5S Quality-gate response vs corruption severity (mean A_quality, COCO val subset)", "",
              "| degradation | clean | s1 | s2 | s3 | s4 | s5 |", "|---|---|---|---|---|---|---|"]
        for dn, rows in res["E5S"].items():
            L.append(f"| {dn} | " + " | ".join(f"{r['mean']:.4f}" for r in rows) + " |")
        L.append("")
    if "E9" in res and "skipped" not in res["E9"]:
        L += ["## E9 AFW level weights (P4 / P8 / P16) by word-height quintile", ""]
        for sname, d in res["E9"].items():
            L += [f"**{sname}**: overall {d['overall']} (std {d['std']})", "", "| height (px) | n | P4 | P8 | P16 |", "|---|---|---|---|---|"]
            for b in d["by_height"]:
                L.append(f"| {b['height'][0]:.0f}-{b['height'][1]:.0f} | {b['n']} | " + " | ".join(f"{x:.4f}" for x in b["w"]) + " |")
            L.append("")
    if "E6" in res:
        L += ["## E6 Fusion-level knockout at test time", "", "| variant | COCO val | IIIT5K |", "|---|---|---|"]
        for g, d in res["E6"].items():
            if isinstance(d, dict): L.append(f"| {g} | {d['coco_val']:.2f} | {d['iiit5k']:.2f} |")
        L.append("")
    if "E8" in res and "skipped" not in res["E8"]:
        L += ["## E8 Per-stride evaluation (levels switched off everywhere, incl. P4 residual)", "",
              "| levels used | COCO val | IIIT5K |", "|---|---|---|"]
        for g, d in res["E8"].items():
            L.append(f"| {g} | {d['coco_val']:.2f} | {d['iiit5k']:.2f} |")
        L.append("")
    if "E7" in res:
        L += ["## E7 QGE attention figure", "", f"![E7]({os.path.basename(res['E7']['figure'])})", ""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--only", default="E1,E2,E3,E4,E5,E5S,E6,E7,E8,E9")
    a = ap.parse_args(); todo = a.only.split(",")
    pre = json.load(open(f"{a.run}/preprocess.json")) if os.path.isfile(f"{a.run}/preprocess.json") else {}
    c = Ctx(a.run, a.limit, pre.get("keep_ratio", True)); cache = {}; res = {}
    meta = {"date": time.strftime("%a %b %d %Y, %I:%M %p"), "img_size": list(c.m.img_size), "keep_ratio": c.kr,
            "device": torch.cuda.get_device_name(0) if c.dev == "cuda" else "cpu", "limit": a.limit}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for e in todo:
        t0 = time.time(); print(f"{e} ...", flush=True)
        if e == "E1": res[e] = E1(c)
        elif e == "E2": res[e] = E2(c, cache)
        elif e == "E3": res[e] = E3(c)
        elif e == "E4": res[e] = E4(c)
        elif e == "E5": res[e] = E5(c, cache)
        elif e == "E6": res[e] = E6(c)
        elif e == "E7": res[e] = E7(c, a.out + "_E7.png", cache)
        elif e == "E8": res[e] = E8(c)
        elif e == "E5S": res[e] = E5S(c)
        elif e == "E9": res[e] = E9(c, cache)
        meta[f"{e}_seconds"] = round(time.time() - t0, 1)
        json.dump({"meta": meta, "results": res}, open(a.out + ".json", "w"), indent=1)
        write_md(a.out + ".md", a.run, res, meta)
    print("done ->", a.out + ".md")


if __name__ == "__main__":
    main()
