"""Per-image comparison of two trained runs (evaluation only). User 2026-10-12.
   python tools/compare_runs.py <run_A> <run_B> <out_prefix>
For COCO-Text v2 val and IIIT5K test: which images A reads correctly and B does not (and the reverse),
overall and per quality quintile (sharpness, crop height, contrast); a PDF with side-by-side examples
of the hardest (least sharp) images where they differ. Writes <out>.md, <out>.json, <out>.pdf."""
import sys, os, json, io
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(__file__))
import ablation_eval as AE
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def main():
    ra, rb, out = sys.argv[1:4]
    na, nb = os.path.basename(ra.rstrip("/")), os.path.basename(rb.rstrip("/"))
    ca = AE.Ctx(ra, None, True); cb = AE.Ctx(rb, None, True); cache = {}
    res, examples = {}, []
    for sname, path in AE.SETS.items():
        pa, pb = ca.preds(path), cb.preds(path)
        q = cache.setdefault(sname, AE.per_image_quality(ca, path))
        oa = np.array([p == g for p, g in pa]); ob = np.array([p == g for p, g in pb])
        d = {"n": len(oa), "acc_A": float(100 * oa.mean()), "acc_B": float(100 * ob.mean()),
             "A_right_B_wrong": int((oa & ~ob).sum()), "A_wrong_B_right": int((~oa & ob).sum()), "bins": {}}
        for key in ("sharpness", "height", "contrast"):
            v = np.array([x[key] for x in q]); e = np.quantile(v, [0, .2, .4, .6, .8, 1]); bins = []
            for b in range(5):
                sel = (v >= e[b]) & ((v <= e[b + 1]) if b == 4 else (v < e[b + 1]))
                bins.append({"range": [float(e[b]), float(e[b + 1])], "n": int(sel.sum()),
                             "acc_A": float(100 * oa[sel].mean()), "acc_B": float(100 * ob[sel].mean()),
                             "A_only": int((oa & ~ob)[sel].sum()), "B_only": int((~oa & ob)[sel].sum())})
            d["bins"][key] = bins
        res[sname] = d
        sh = np.array([x["sharpness"] for x in q])
        w = AE.Wrap(ca.ds(path))
        for kind, mask in (("A right, B wrong", oa & ~ob), ("A wrong, B right", ~oa & ob)):
            idx = [int(i) for i in np.argsort(sh) if mask[i]][:24]
            for i in idx:
                p, j = w.idx[i]
                examples.append((sname, kind, p.raw(j), pa[i][1], pa[i][0], pb[i][0], float(sh[i])))
        print(f"{sname}: A {d['acc_A']:.2f} B {d['acc_B']:.2f} A-only {d['A_right_B_wrong']} B-only {d['A_wrong_B_right']}", flush=True)
    json.dump(res, open(out + ".json", "w"), indent=1)
    L = [f"# Per-image comparison: A = `{na}` vs B = `{nb}`", ""]
    for sname, d in res.items():
        L += [f"## {sname} ({d['n']} images): A {d['acc_A']:.2f}%, B {d['acc_B']:.2f}%; "
              f"A right & B wrong {d['A_right_B_wrong']}, A wrong & B right {d['A_wrong_B_right']}", ""]
        for key, bins in d["bins"].items():
            L += [f"**by {key}** (Q1 = lowest)", "", "| quintile | range | n | A % | B % | A - B | A only | B only |", "|---|---|---|---|---|---|---|---|"]
            for k, b in enumerate(bins, 1):
                L.append(f"| Q{k} | {b['range'][0]:.0f}-{b['range'][1]:.0f} | {b['n']} | {b['acc_A']:.2f} | {b['acc_B']:.2f} | "
                         f"{b['acc_A'] - b['acc_B']:+.2f} | {b['A_only']} | {b['B_only']} |")
            L.append("")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    c = canvas.Canvas(out + ".pdf", pagesize=A4); W, H = A4
    c.setFont("Helvetica-Bold", 14); c.drawString(40, H - 50, f"A = {na}   vs   B = {nb}")
    c.setFont("Helvetica", 10); c.drawString(40, H - 68, "Least-sharp images where the two models differ (24 per set and direction).")
    y = H - 95
    for sname, kind, img, gt, pa_, pb_, s in examples:
        if y < 70:
            c.showPage(); y = H - 50
        h, w_ = img.shape[:2]; sc = min(150 / w_, 40 / h)
        im = cv2.resize(img, (max(1, int(w_ * sc)), max(1, int(h * sc))))
        buf = io.BytesIO(); buf.write(cv2.imencode(".png", im)[1].tobytes()); buf.seek(0)
        c.drawImage(ImageReader(buf), 40, y - 40, im.shape[1], im.shape[0])
        c.setFont("Helvetica", 8); c.drawString(200, y - 8, f"{sname} | {kind} | sharpness {s:.0f}")
        c.setFont("Courier-Bold", 10); c.drawString(200, y - 22, f"GT: {gt}")
        c.setFillColorRGB(0, 0.5, 0) if pa_ == gt else c.setFillColorRGB(0.8, 0, 0); c.drawString(200, y - 35, f"A : {pa_}")
        c.setFillColorRGB(0, 0.5, 0) if pb_ == gt else c.setFillColorRGB(0.8, 0, 0); c.drawString(380, y - 35, f"B : {pb_}")
        c.setFillColorRGB(0, 0, 0); y -= 52
    c.save(); print("done ->", out + ".md")


if __name__ == "__main__":
    main()
