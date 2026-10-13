import sys, json, io, collections, torch, cv2, numpy as np, lmdb
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import train_textnet as T
from qacspnet.models import QACSPTextNet
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
RUN = sys.argv[1]; OUT = sys.argv[2]
import csv
NAMES = [r["ImgName"] for r in csv.DictReader(open("/home/deepak/Documents/STR-DATASET/iiit5k/testdata.csv"))]
def main():
    torch.set_num_threads(8)
    cfg = json.load(open(f"outputs/{RUN}/model_config.json")); pre = json.load(open(f"outputs/{RUN}/preprocess.json"))
    m = QACSPTextNet(**cfg); m.load_state_dict(torch.load(f"outputs/{RUN}/best.pth", map_location="cpu")); m.eval()
    path = "/home/deepak/Documents/qacspnet-TRAIN/ext_data/iiit5k_test"
    ds = T.build_dataset("/home/deepak/Documents/qacspnet-TRAIN/ext_data", "iiit5k_test", None, 25, m.img_size); ds.keep_ratio = pre["keep_ratio"]
    dl = torch.utils.data.DataLoader(ds, 250, collate_fn=T.collate, num_workers=0)
    res = []
    with torch.no_grad():
        for imgs, texts in dl:
            for p, t in zip(m.recognize(imgs), texts):
                g = "".join(c for c in m.normalize(t) if c in m.alphabet); res.append((g, p))
    errs = [(i, g, p) for i, (g, p) in enumerate(res) if g != p]
    acc = 1 - len(errs) / len(res)
    kinds = collections.Counter("same length (substitution)" if len(p) == len(g) else "shorter (missing chars)" if len(p) < len(g) else "longer (extra chars)" for _, g, p in errs)
    txn = T._open_lmdb(path).begin()
    c = canvas.Canvas(OUT, pagesize=A4); W, H = A4
    # title page
    c.setFont("Helvetica-Bold", 16); c.drawString(50, H - 70, f"IIIT5K test errors: {RUN}")
    c.setFont("Helvetica", 11); y = H - 100
    for line in [f"Model: QA-CSPTextNet-S, checkpoint outputs/{RUN}/best.pth",
                 f"Input: {m.img_size[0]}x{m.img_size[1]}, keep-ratio={pre['keep_ratio']}, grayscale, greedy CTC decoding",
                 f"Images: {len(res)}   correct: {len(res) - len(errs)}   errors: {len(errs)}   accuracy: {100 * acc:.2f}%",
                 "", "Error types:"] + [f"   {k}: {v}" for k, v in kinds.most_common()] + [
                 "", "Each row: original test image (left), model input view (48x192, keep-ratio), GT and prediction.",
                 "Differing characters in the prediction are marked in red (character-level alignment)."]:
        c.drawString(50, y, line); y -= 16
    c.showPage()
    import difflib
    ROW = 62; per_page = int((H - 90) // ROW); n = 0
    for k, (i, g, p) in enumerate(errs):
        if n % per_page == 0:
            if n: c.showPage()
            c.setFont("Helvetica-Bold", 10); c.drawString(50, H - 40, f"{RUN}: errors {n + 1}-{min(n + per_page, len(errs))} of {len(errs)}")
            c.setFont("Helvetica", 9); c.drawString(50, H - 54, f"#   original image (file name)         model input ({m.img_size[0]}x{m.img_size[1]})              GT  ->  prediction")
        y = H - 80 - (n % per_page) * ROW
        key = ds.index[i]
        raw = cv2.imdecode(np.frombuffer(txn.get(f"image-{key:09d}".encode()), np.uint8), cv2.IMREAD_COLOR)
        h0, w0 = raw.shape[:2]; s = min(150 / w0, 44 / h0); disp = cv2.resize(raw, (max(1, int(w0 * s)), max(1, int(h0 * s))))
        ok, buf = cv2.imencode(".png", disp); c.drawImage(ImageReader(io.BytesIO(buf.tobytes())), 70, y - 44 + (44 - disp.shape[0]) / 2, width=disp.shape[1], height=disp.shape[0])
        inp = ((ds[i][0][0].numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", inp); c.drawImage(ImageReader(io.BytesIO(buf.tobytes())), 230, y - 44 + 4, width=160, height=40)
        c.setFont("Helvetica", 8); c.setFillColorRGB(0, 0, 0); c.drawString(50, y - 20, str(k + 1)); c.drawString(70, y - 52, f"{NAMES[key - 1]}  ({w0}x{h0}px)")
        c.setFont("Courier-Bold", 11 if len(g) <= 14 else max(6, int(11 * 14 / len(g)))); c.drawString(400, y - 14, f"GT:   {g}")
        fs = 11 if max(len(g), len(p)) <= 14 else max(6, int(11 * 14 / max(len(g), len(p))))
        c.setFont("Courier-Bold", fs); x = 400; c.drawString(x, y - 32, "pred: "); x += c.stringWidth("pred: ", "Courier-Bold", fs)
        sm = difflib.SequenceMatcher(None, g, p)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            seg = p[j1:j2]
            if tag == "equal": c.setFillColorRGB(0, 0, 0)
            else: c.setFillColorRGB(0.85, 0, 0)
            c.drawString(x, y - 32, seg); x += c.stringWidth(seg, "Courier-Bold", fs)
            if tag == "delete": c.setFillColorRGB(0.85, 0, 0); c.drawString(x, y - 32, "_" * (i2 - i1)); x += c.stringWidth("_" * (i2 - i1), "Courier-Bold", fs)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 8)
        c.drawString(400, y - 46, "same length" if len(p) == len(g) else ("missing chars" if len(p) < len(g) else "extra chars"))
        c.setStrokeColorRGB(0.85, 0.85, 0.85); c.line(50, y - 56, W - 50, y - 56)
        n += 1
    c.save(); print(f"{len(errs)} errors, accuracy {100*acc:.2f}% -> {OUT}")
if __name__ == "__main__": main()
