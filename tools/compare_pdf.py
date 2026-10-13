#!/usr/bin/env python3
# Side-by-side IIIT5K error comparison of two runs (CPU).
#   python tools/compare_pdf.py s7_afwsize s9_dataaug reports/compare_s7_s9.pdf
# Rows = every test image that at least one of the two models gets wrong, grouped as
# FIXED (A wrong, B right), NEW ERROR (A right, B wrong), STILL WRONG (both wrong).
import difflib, io, json, sys
import cv2, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import train_textnet as T
from qacspnet.models import QACSPTextNet
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def predict(run):
    cfg = json.load(open(f"outputs/{run}/model_config.json")); pre = json.load(open(f"outputs/{run}/preprocess.json"))
    m = QACSPTextNet(**cfg); m.load_state_dict(torch.load(f"outputs/{run}/best.pth", map_location="cpu")); m.eval()
    ds = T.build_dataset("/home/deepak/Documents/qacspnet-TRAIN/ext_data", "iiit5k_test", None, 25, m.img_size)
    ds.keep_ratio = pre.get("keep_ratio", False); ds.color = pre.get("in_chans", 1) == 3
    out = []
    with torch.no_grad():
        for imgs, texts in torch.utils.data.DataLoader(ds, 250, collate_fn=T.collate, num_workers=0):
            for p, t in zip(m.recognize(imgs), texts):
                out.append(("".join(c for c in m.normalize(t) if c in m.alphabet), p))
    return out, ds, float(open(f"outputs/{run}/best_score.txt").read())

def png(img):
    ok, buf = cv2.imencode(".png", img); return ImageReader(io.BytesIO(buf.tobytes()))

def fsize(t):
    return 10 if len(t) <= 14 else max(6, int(10 * 14 / len(t)))

def draw_pred(c, x, y, gt, pred, correct):
    fs = fsize(max(gt, pred, key=len)); c.setFont("Courier-Bold", fs)
    if correct:
        c.setFillColorRGB(0, 0.5, 0); c.drawString(x, y, pred); c.setFillColorRGB(0, 0, 0); return
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, pred).get_opcodes():
        c.setFillColorRGB(0, 0, 0) if tag == "equal" else c.setFillColorRGB(0.85, 0, 0)
        seg = pred[j1:j2] if tag != "delete" else "_" * (i2 - i1)
        c.drawString(x, y, seg); x += c.stringWidth(seg, "Courier-Bold", fs)
    c.setFillColorRGB(0, 0, 0)

def main():
    A, B, OUT = sys.argv[1:4]
    torch.set_num_threads(8)
    ra, ds, sa = predict(A); rb, _, sb = predict(B)
    groups = {"FIXED": [], "NEW ERROR": [], "STILL WRONG": []}
    for i, ((g, pa), (_, pb)) in enumerate(zip(ra, rb)):
        oka, okb = pa == g, pb == g
        if not oka and okb: groups["FIXED"].append(i)
        elif oka and not okb: groups["NEW ERROR"].append(i)
        elif not oka and not okb: groups["STILL WRONG"].append(i)
    txn = T._open_lmdb(ds.path).begin()
    c = canvas.Canvas(OUT, pagesize=A4); W, H = A4
    c.setFont("Helvetica-Bold", 16); c.drawString(50, H - 70, f"IIIT5K test: {A} vs {B}")
    c.setFont("Helvetica", 11); y = H - 100
    for line in [f"{A}: {100*sa:.2f}%  ({sum(p != g for g, p in ra)} errors)",
                 f"{B}: {100*sb:.2f}%  ({sum(p != g for g, p in rb)} errors)", "",
                 f"FIXED        (wrong in {A}, correct in {B}):  {len(groups['FIXED'])}",
                 f"NEW ERROR    (correct in {A}, wrong in {B}):  {len(groups['NEW ERROR'])}",
                 f"STILL WRONG  (wrong in both):                 {len(groups['STILL WRONG'])}", "",
                 "Columns: original image | GT | " + A + " prediction | " + B + " prediction.",
                 "Green = correct; red characters = where the prediction differs from the GT.",
                 "Sections in order: FIXED, NEW ERROR, STILL WRONG."]:
        c.drawString(50, y, line); y -= 16
    c.showPage()
    ROW = 50; per = int((H - 100) // ROW)
    colx = {"img": 45, "gt": 215, "a": 330, "b": 450}
    for title, rows in groups.items():
        for k in range(0, max(1, len(rows)), per):
            chunk = rows[k:k + per]
            if not chunk and k: break
            color = {"FIXED": (0, 0.5, 0), "NEW ERROR": (0.85, 0, 0), "STILL WRONG": (0.4, 0.4, 0.4)}[title]
            c.setFillColorRGB(*color); c.setFont("Helvetica-Bold", 12)
            c.drawString(45, H - 40, f"{title}: {k + 1}-{k + len(chunk)} of {len(rows)}"); c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 9)
            for key, name in [("img", "image"), ("gt", "GT"), ("a", A), ("b", B)]: c.drawString(colx[key], H - 58, name)
            for n, i in enumerate(chunk):
                y = H - 70 - n * ROW
                raw = cv2.imdecode(np.frombuffer(txn.get(f"image-{ds.index[i]:09d}".encode()), np.uint8), cv2.IMREAD_COLOR)
                h0, w0 = raw.shape[:2]; s = min(160 / w0, 40 / h0)
                disp = cv2.resize(raw, (max(1, int(w0 * s)), max(1, int(h0 * s))))
                c.drawImage(png(disp), colx["img"], y - 42 + (40 - disp.shape[0]) / 2, width=disp.shape[1], height=disp.shape[0])
                g = ra[i][0]
                c.setFont("Courier-Bold", fsize(g)); c.drawString(colx["gt"], y - 24, g)
                draw_pred(c, colx["a"], y - 24, g, ra[i][1], ra[i][1] == g)
                draw_pred(c, colx["b"], y - 24, g, rb[i][1], rb[i][1] == g)
                c.setFont("Helvetica", 7); c.setFillColorRGB(0.5, 0.5, 0.5); c.drawString(colx["gt"], y - 36, f"#{i}  {w0}x{h0}px"); c.setFillColorRGB(0, 0, 0)
                c.setStrokeColorRGB(0.85, 0.85, 0.85); c.line(45, y - 46, W - 45, y - 46)
            c.showPage()
    c.save()
    print(f"{A} {100*sa:.2f}% vs {B} {100*sb:.2f}%: fixed {len(groups['FIXED'])}, new {len(groups['NEW ERROR'])}, still wrong {len(groups['STILL WRONG'])} -> {OUT}")

if __name__ == "__main__": main()
