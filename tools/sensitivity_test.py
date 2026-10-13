import sys, json, torch, numpy as np, cv2
sys.path.insert(0, "/home/deepak/Documents/qacspnet-TRAIN"); sys.path.insert(0, "/home/deepak/Documents/qacspnet-TRAIN/tools")
import train_textnet as T
from qacspnet.models import QACSPTextNet
class Fixed:  # rng stand-in with fixed uniform value, deterministic choices
    def __init__(s, u, r=0.9): s.u, s.r = u, r
    def random(s): return s.r
    def uniform(s, a, b): return s.u
    def integers(s, a, b=None, size=None): return np.random.default_rng(0).integers(a, b, size)
    def choice(s, v, *a, **k): return v[0] if hasattr(v, "__len__") else v
    def normal(s, *a, **k): return np.random.default_rng(0).normal(*a, **k)
def arc(img, span, smile):
    aug = T.CurveSymbolAugment(1, 0)
    class R(Fixed):
        def __init__(s): super().__init__(span); s.c = 0
        def random(s): s.c += 1; return (0.9 if not smile else 0.1) if s.c == 1 else 0.5
    return aug.arc(img, R())
def slant(img, sh):
    h, w = img.shape[:2]; pad = int(abs(sh) * h) + 2
    im = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_REPLICATE); W = im.shape[1]
    M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]]); return cv2.warpAffine(im, M, (W, h), borderMode=cv2.BORDER_REPLICATE)
def rotate(img, ang):
    h, w = img.shape[:2]; pad = int(0.3 * h) + 2
    im = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE); H, W = im.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0); return cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
def edge_noise(img):
    h, w = img.shape[:2]; rng = np.random.default_rng(1)
    strip = cv2.GaussianBlur(rng.integers(0, 256, (h, max(3, w // 8))).astype(np.uint8), (0, 0), 1.0)
    return np.concatenate([img, strip], 1)
def dots(img):
    out = img.copy(); h, w = out.shape[:2]; rng = np.random.default_rng(2)
    bg = float(np.median(np.concatenate([out[0], out[-1]]))); val = 20 if bg > 128 else 235
    for _ in range(2): cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(int(0.7 * h), h))), max(1, h // 10), val, -1)
    return out
def exclaim(img):
    out = img.copy(); h, w = out.shape[:2]; bg = float(np.median(np.concatenate([out[0], out[-1]]))); val = 20 if bg > 128 else 235
    out = cv2.copyMakeBorder(out, 0, 0, 0, max(4, w // 10), cv2.BORDER_REPLICATE); W = out.shape[1]; t = max(1, h // 12); x = W - max(4, w // 10) + t
    cv2.rectangle(out, (x, int(0.1 * h)), (x + t, int(0.7 * h)), val, -1); cv2.circle(out, (x + t // 2, int(0.88 * h)), max(1, int(t * 0.8)), val, -1); return out
def main():
    torch.set_num_threads(8)
    cfg = json.load(open("outputs/s7_afwsize/model_config.json"))
    m = QACSPTextNet(**cfg); m.load_state_dict(torch.load("outputs/s7_afwsize/best.pth", map_location="cpu")); m.eval()
    ds = T.build_dataset("data_cache", "iiit5k_train.keys.json", m.alphabet, 25, m.img_size).datasets[0]
    idx = list(range(0, 2000, 2))  # 1000 crops
    raws = [(ds.raw(i), T._open_lmdb(ds.path).begin().get(f"label-{ds.index[i]:09d}".encode()).decode().strip().lower()) for i in idx]
    def acc(fn):
        imgs = torch.stack([T._chw(T.fit_keep_ratio(fn(r), m.img_size)) for r, _ in raws]); ok = 0
        with torch.no_grad():
            for s in range(0, len(imgs), 250):
                for p, (_, g) in zip(m.recognize(imgs[s:s + 250]), raws[s:s + 250]): ok += (p == g)
        return 100 * ok / len(raws)
    tests = [("straight (reference)", lambda x: x), ("arc 30 deg", lambda x: arc(x, 30, False)), ("arc 60 deg", lambda x: arc(x, 60, False)),
             ("arc 90 deg", lambda x: arc(x, 90, False)), ("arc 120 deg", lambda x: arc(x, 120, False)), ("arc 90 deg smile", lambda x: arc(x, 90, True)),
             ("slant 0.25", lambda x: slant(x, 0.25)), ("slant 0.5", lambda x: slant(x, 0.5)), ("rotate 8 deg", lambda x: rotate(x, 8)), ("rotate 15 deg", lambda x: rotate(x, 15)),
             ("edge noise strip (right)", edge_noise), ("2 dots at baseline", dots), ("exclamation mark at end", exclaim)]
    print(f"{'distortion':28s} {'S7 acc on IIIT5K-train (1000)':>30s}")
    for name, fn in tests: print(f"{name:28s} {acc(fn):29.1f}%", flush=True)
if __name__ == "__main__": main()
