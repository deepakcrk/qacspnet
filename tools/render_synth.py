#!/usr/bin/env python3
# On-the-fly synthetic word renderer for QA-CSPTextNet training (used via a
# data_cache/render.keys.json stub: {"render": true, "n": 200000}).
# Content: numbers (1-6 digits, short ones weighted up), lexicon words, random alnum strings,
# with symbols ($ Rs. % # ! ? . , - brackets ...) mixed in; the LABEL is the alnum core.
# Look: 300+ system fonts (thick strokes via stroke_width), per-character colours, gradients,
# noise textures, shadows / outlines, illumination changes. Geometry (arc / bend, slant,
# rotation), curved lines, boxes, scratches, edge clutter etc. come from CurveSymbolAugment.
import glob
import json
import os
import random
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts"]
BAD = ("mathjax", "cm", "msam", "msbm", "esint", "eufm", "dsrom", "d050000l", "rsfs", "wasy",
       "stmary", "symbol", "dingbat", "emoji", "music", "braille", "math")
SYMS = list("$%#@&!?.,:;-_/\\()[]{}<>'\"*+=~^|") + ["Rs", "Rs.", "₹", "€", "£", "...", "--", "!!",
                                                     "()", "[]", "{}", "''", "\"\""]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache")


def _font_ok(path):
    low = os.path.basename(path).lower()
    if any(b in low for b in BAD):
        return False
    try:
        f = ImageFont.truetype(path, 40)
        def draw(ch):
            im = Image.new("L", (90, 90), 0)
            ImageDraw.Draw(im).text((10, 10), ch, font=f, fill=255)
            return np.array(im)
        # fonts without Latin glyphs draw the same "tofu" box for every character
        for a_, b_ in (("A", "B"), ("o", "l"), ("1", "7"), ("g", "x")):
            ia, ib = draw(a_), draw(b_)
            if ia.sum() < 2000 or ib.sum() < 2000 or np.array_equal(ia, ib):
                return False
        return True
    except Exception:
        return False


FANCY_DIR = "/home/deepak/Documents/STR-DATASET/fonts/google/ofl"


def _font_full_ok(path):
    """All of a-z / 0-9 drawable and mutually distinct (no tofu, no missing glyphs)."""
    try:
        f = ImageFont.truetype(path, 40)
        seen = set()
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
            im = Image.new("L", (90, 90), 0)
            ImageDraw.Draw(im).text((10, 10), ch, font=f, fill=255)
            a = np.array(im)
            if a.sum() < 1500:
                return False
            seen.add(a.tobytes())
        return len(seen) >= 34
    except Exception:
        return False


def fancy_font_list():
    cache = os.path.join(CACHE, "fonts_fancy.json")
    if os.path.isfile(cache):
        return json.load(open(cache))
    paths = sorted(glob.glob(os.path.join(FANCY_DIR, "*", "*.ttf")))
    ok = [p for p in paths if _font_full_ok(p)]
    json.dump(ok, open(cache, "w"))
    return ok


def font_list():
    cache = os.path.join(CACHE, "fonts.json")
    if os.path.isfile(cache):
        return json.load(open(cache))
    paths = []
    for d in FONT_DIRS:
        paths += glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)
        paths += glob.glob(os.path.join(d, "**", "*.otf"), recursive=True)
    ok = sorted(p for p in paths if _font_ok(p))
    json.dump(ok, open(cache, "w"))
    return ok


class Renderer:
    LEN_P = np.array([6, 9, 11, 12, 12, 11, 9, 8, 6, 5, 4], float)  # lengths 1..11
    LEN_P /= LEN_P.sum()

    def __init__(self, lexicon=None, color=True, p_fancy=0.25):
        self.fonts = font_list()
        self.fancy = fancy_font_list() if os.path.isdir(FANCY_DIR) else []
        self.p_fancy = p_fancy
        self.lexicon = lexicon or []
        self.by_len = {}
        for w in self.lexicon:
            self.by_len.setdefault(len(w), []).append(w)
        self.color = color
        self._patches = None  # tiny strips of real images (data_cache/bg_patches.pkl), lazy

    def word_len(self, rng):
        if rng.random() < 0.03:  # rare long words (12-20)
            return int(rng.integers(12, 21))
        return int(rng.choice(np.arange(1, 12), p=self.LEN_P))
        self._patches = None  # tiny strips of real images (data_cache/bg_patches.pkl), lazy

    def patches(self):
        if self._patches is None:
            import pickle

            f = os.path.join(CACHE, "bg_patches.pkl")
            self._patches = pickle.load(open(f, "rb")) if os.path.isfile(f) else []
        return self._patches

    def full_images(self):
        if getattr(self, "_full", None) is None:
            import pickle

            f = os.path.join(CACHE, "bg_full.pkl")
            self._full = pickle.load(open(f, "rb")) if os.path.isfile(f) else []
        return self._full

    def full_background(self, w, h, rng):
        """A large region (30-100%) of a whole real image, resized DOWN to the canvas, so the
        clutter / texture / noise of the scene stays in the background."""
        pool = self.full_images()
        ar = w / max(1, h)
        for _ in range(12):  # find a region that contains no real text (COCO-Text / plates)
            item = pool[int(rng.integers(len(pool)))]
            buf, boxes = item if isinstance(item, tuple) else (item, [])
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            H0, W0 = img.shape[:2]
            for _ in range(6):
                f = rng.uniform(0.3, 1.0)
                ch, cw = max(8, int(H0 * f)), max(8, int(W0 * f))
                if cw / ch > ar:  # keep roughly the canvas aspect so the downscale is not a stretch
                    cw = max(8, min(W0, int(ch * ar)))
                else:
                    ch = max(8, min(H0, int(cw / ar)))
                y0, x0 = int(rng.integers(0, max(1, H0 - ch + 1))), int(rng.integers(0, max(1, W0 - cw + 1)))
                sy = h / ch  # scale of this crop onto the canvas
                # text inside is fine when it becomes unreadable (< 6 px high on the canvas)
                if all(x0 + cw <= b[0] or x0 >= b[2] or y0 + ch <= b[1] or y0 >= b[3]
                       or (b[3] - b[1] - 6) * sy < 6 for b in boxes):
                    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_AREA)
        return self.patch_background(w, h, rng)  # no text-free region found

    def patch_background(self, w, h, rng):
        """1-3 tiny real-image strips, each stretched to a segment of the canvas (different
        texture / colour per segment), with soft blended seams."""
        pool = self.patches()
        n = int(rng.integers(1, 4))
        vertical = rng.random() < 0.6  # segments side by side (else stacked)
        L = w if vertical else h
        cuts = sorted(rng.integers(1, max(2, L), n - 1).tolist()) if n > 1 else []
        bounds = [0] + cuts + [L]
        bg = np.zeros((h, w, 3), np.float32)
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b <= a:
                continue
            p = pool[int(rng.integers(len(pool)))]
            seg = cv2.resize(p, (b - a, h) if vertical else (w, b - a), interpolation=cv2.INTER_LINEAR)
            if rng.random() < 0.5:  # brightness / tint change per segment
                seg = np.clip(seg.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.integers(-40, 41, 3), 0, 255)
            if vertical:
                bg[:, a:b] = seg
            else:
                bg[a:b, :] = seg
        if n > 1:  # soften the seams
            k = max(3, int(rng.uniform(0.02, 0.08) * L)) | 1
            bg = cv2.GaussianBlur(bg, (k, 1) if vertical else (1, k), 0)
        return np.clip(bg, 0, 255).astype(np.uint8)

    # ---------------- content ----------------
    def text(self, rng):
        kind = rng.random()
        n = self.word_len(rng)
        if kind < 0.4:  # numbers (1-10 digits)
            core = "".join(str(int(rng.integers(10))) for _ in range(min(n, 10)))
        elif kind < 0.8 and self.by_len.get(n):
            bucket = self.by_len[n]
            core = bucket[int(rng.integers(len(bucket)))]
        else:
            core = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789")) for _ in range(n))
        case = rng.random()
        shown = core.upper() if case < 0.6 else core.title() if case < 0.75 else core
        if rng.random() < 0.6:  # mix in symbols; the label stays the alnum core
            chars = list(shown)
            for _ in range(int(rng.integers(1, 4))):
                sym = SYMS[int(rng.integers(len(SYMS)))]
                pos = int(rng.integers(0, len(chars) + 1))
                chars.insert(pos, sym)
            shown = "".join(chars)
        return shown, core.lower()

    # ---------------- background ----------------
    def add_noise(self, bg, rng):
        """Grain, blotches or streaks so the background is rarely clean."""
        h, w = bg.shape[:2]
        out = bg.astype(np.float32)
        kind = rng.random()
        if kind < 0.4:  # fine grain
            out += rng.normal(0, rng.uniform(3, 12), out.shape)
        elif kind < 0.7:  # blotches (low-frequency)
            n = rng.normal(0, 1, (max(2, h // 8), max(2, w // 8), 3)).astype(np.float32)
            n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * rng.uniform(8, 25)
            out += n
        else:  # streaks / scratches of texture
            n = rng.normal(0, 1, (h, max(2, w // 16), 3)).astype(np.float32) if rng.random() < 0.5 else rng.normal(0, 1, (max(2, h // 6), w, 3)).astype(np.float32)
            out += cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR) * rng.uniform(6, 20)
        return np.clip(out, 0, 255).astype(np.uint8)

    def background(self, w, h, rng):
        if not getattr(self, "_simple", False) and rng.random() < 0.5:  # clean background
            self._last_bg_kind = "clean"
            base = rng.integers(0, 256, 3)
            bg = np.empty((h, w, 3), np.uint8); bg[:] = base
            if rng.random() < 0.4:  # gentle gradient
                t = np.linspace(-1, 1, w)[None, :, None] * rng.uniform(5, 25)
                bg = np.clip(bg.astype(np.float32) + t, 0, 255).astype(np.uint8)
            return bg
        if getattr(self, "_simple", False):
            self._last_bg_kind = "simple"
            bg = np.empty((h, w, 3), np.uint8); bg[:] = rng.integers(0, 256, 3)
            return bg
        bg = self._background(w, h, rng)
        return self.add_noise(bg, rng) if rng.random() < 0.5 else bg

    def _background(self, w, h, rng):
        r = rng.random()
        if self.full_images() and r < 0.3:
            self._last_bg_kind = "full"
            return self.full_background(w, h, rng)
        self._last_bg_kind = "other"
        if self.patches() and r < 0.6:
            return self.patch_background(w, h, rng)
        kind = rng.random()
        base = rng.integers(0, 256, 3)
        if kind < 0.4:
            bg = np.empty((h, w, 3), np.uint8)
            bg[:] = base
        elif kind < 0.7:  # gradient
            a, b = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            t = np.linspace(0, 1, w if rng.random() < 0.5 else h)
            g = (a[None] * (1 - t[:, None]) + b[None] * t[:, None]).astype(np.uint8)
            bg = np.tile(g[None], (h, 1, 1)) if len(t) == w else np.tile(g[:, None], (1, w, 1))
        else:  # noise texture
            bg = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
            bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(1, 6))
            bg = np.clip((bg.astype(np.int16) - 128) * rng.uniform(0.15, 0.5) + base, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(bg)

    SIGN_PAIRS = [((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0)), ((10, 10, 10), (0, 220, 255)),
                  ((0, 210, 255), (10, 10, 10)), ((160, 60, 10), (255, 255, 255)), ((30, 30, 190), (255, 255, 255)),
                  ((40, 120, 20), (255, 255, 255)), ((255, 255, 255), (30, 30, 200))]  # (background, text)

    def _sign_pair(self, h, w, rng):
        """A classic sign colour pair; light speckle on the background only."""
        bg_c, fg = self.SIGN_PAIRS[int(rng.integers(len(self.SIGN_PAIRS)))]
        jitter = rng.integers(-15, 16, 3)
        bg = np.empty((h, w, 3), np.uint8); bg[:] = np.clip(np.array(bg_c) + jitter, 0, 255)
        if rng.random() < 0.4:
            bg = np.clip(bg.astype(np.int16) + rng.normal(0, rng.uniform(3, 10), bg.shape), 0, 255).astype(np.uint8)
        return bg, tuple(int(v) for v in np.clip(np.array(fg) + rng.integers(-10, 11, 3), 0, 255))

    @staticmethod
    def _contrast_color(bg_mean, rng):
        lum = 0.299 * bg_mean[2] + 0.587 * bg_mean[1] + 0.114 * bg_mean[0]
        if lum > 128:
            c = rng.integers(0, 90, 3)
        else:
            c = rng.integers(170, 256, 3)
        if rng.random() < 0.5:  # saturate one channel for coloured text
            c[int(rng.integers(3))] = rng.integers(0, 256)
        return tuple(int(v) for v in c)

    # ---------------- render ----------------
    MIN_CONTRAST = 48  # LAB distance between text pixels and their surroundings

    def render(self, rng=None):
        rng = rng or np.random.default_rng()
        for _ in range(6):  # re-draw when the text ends up illegible against its background
            out, label, ok = self._render(rng)
            if ok and self._big_enough(out):
                if rng.random() < getattr(self, "p_object", 0.35):
                    out = self._add_object(out, rng)
                return out, label
        self._simple = True  # last resort: plain background, solid paint (always readable)
        try:
            out, label, _ = self._render(rng)
        finally:
            self._simple = False
        return out, label

    def _arc_text(self, rng, shown, font, size, stroke):
        """Each character drawn rotated along a circular arc on a FLAT background.
        Kinds: centred rainbow / smile, or one part of a circle (falling / rising)."""
        asc, desc = font.getmetrics()
        adv = []
        for ch in shown:
            try:
                adv.append(max(2.0, font.getlength(ch)) + stroke)
            except Exception:
                adv.append(size * 0.6)
        L = float(sum(adv))
        kind = rng.random()
        below = rng.random() < 0.4  # smile-like: letters on the lower part of a circle
        if kind < 0.55:  # centred arc
            span = np.radians(rng.uniform(30, 110))
            R = max(L / span, 1.6 * (asc + desc))
            span = L / R
            a0 = -span / 2
        else:  # one part of a circle
            span = np.radians(rng.uniform(45, 110))
            R = max(L / span, 1.6 * (asc + desc))
            span = L / R
            a0 = np.radians(rng.uniform(-15, 10)) if rng.random() < 0.6 else -span + np.radians(rng.uniform(-10, 15))
        S = int(2.2 * (asc + desc) + 2 * stroke + 4)
        per_color, per_thick = rng.random() < 0.5, rng.random() < 0.3
        placed, cum = [], 0.0
        for ch, a in zip(shown, adv):
            th_ = a0 + (cum + a / 2) / R
            cum += a
            if not below:  # top of the circle, letters point outwards
                px, py = R * np.sin(th_), -R * np.cos(th_)
                rot = -np.degrees(th_)
            else:  # bottom of the circle (reading left to right), letters point to the centre
                px, py = R * np.sin(-th_ + 0) * -1, R * np.cos(th_)
                rot = np.degrees(th_)
            placed.append((ch, px, py, rot))
        xs = [p[1] for p in placed]; ys = [p[2] for p in placed]
        m = S // 2 + 4
        x_min, y_min = min(xs) - m, min(ys) - m
        W, H = int(max(xs) - x_min + m), int(max(ys) - y_min + m)
        if rng.random() < 0.35:
            bg, color = self._sign_pair(H, W, rng); self._last_bg_kind = "clean"
            per_color = False
        else:
            bg = self.background(W, H, rng)
            color = self._contrast_color(bg.reshape(-1, 3).mean(0), rng)
        bg_mean = bg.reshape(-1, 3).mean(0)
        canvas = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
        mask = Image.new("L", (W, H), 0)
        for ch, px, py, rot in placed:
            c = self._contrast_color(bg_mean, rng) if (per_color and rng.random() < 0.7) else color
            sw = int(rng.choice([0, 1, 2, 3, 4])) if per_thick else stroke
            tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            try:
                cw = font.getlength(ch)
            except Exception:
                cw = size * 0.6
            ImageDraw.Draw(tile).text((S / 2 - cw / 2, S / 2 - asc), ch, font=font, fill=tuple(c) + (255,), stroke_width=sw)
            tile = tile.rotate(rot, resample=Image.BICUBIC, center=(S / 2, S / 2))
            pos = (int(round(px - x_min - S / 2)), int(round(py - y_min - S / 2)))
            canvas.alpha_composite(tile, dest=(max(0, pos[0]), max(0, pos[1])), source=(max(0, -pos[0]), max(0, -pos[1])))
            mask.paste(tile.split()[3], pos, tile.split()[3])
        out = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        mk = np.array(mask) > 127
        ys_, xs_ = np.where(mk)
        if len(xs_):  # crop to the text with a little margin of background
            pad = int(0.15 * size) + 2
            y0, y1 = max(0, ys_.min() - pad), min(H, ys_.max() + pad)
            x0, x1 = max(0, xs_.min() - pad), min(W, xs_.max() + pad)
            out, bg, mk = out[y0:y1, x0:x1], bg[y0:y1, x0:x1], mk[y0:y1, x0:x1]
        return out, bg, mk

    def _add_object(self, out, rng):
        """A small object next to the text (inside the image, not touching it): a small piece of
        a real photo or a drawn leaf / flower / star / heart, on the image's own background."""
        h, w = out.shape[:2]
        sz = max(6, int(h * rng.uniform(0.35, 0.8)))
        gap = max(1, int(h * rng.uniform(0.05, 0.25)))
        strip = np.empty((h, sz + gap + max(1, int(0.1 * h))) + out.shape[2:], np.uint8)
        edge = out[:, -1] if rng.random() < 0.5 else out[:, 0]
        strip[:] = np.median(edge, axis=0).astype(np.uint8)
        y = int(rng.integers(0, max(1, h - sz + 1)))
        x = gap
        if self.coco_images():  # a piece of a real COCO photo, wiggly outline (no drawn shapes)
            if self._paste_blob(strip, x + sz // 2, y + sz // 2, sz, rng):
                return np.concatenate([out, strip], axis=1) if rng.random() < 0.65 else np.concatenate([strip[:, ::-1], out], axis=1)
        if False:
            item = self.full_images()[int(rng.integers(len(self.full_images())))]
            buf = item[0] if isinstance(item, tuple) else item
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            H0, W0 = img.shape[:2]; cs = int(min(H0, W0) * rng.uniform(0.1, 0.3))
            y0, x0 = int(rng.integers(0, H0 - cs)), int(rng.integers(0, W0 - cs))
            obj = cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (sz, sz), interpolation=cv2.INTER_AREA)
            if out.ndim == 2:
                obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            if rng.random() < 0.5:  # round object
                m = np.zeros((sz, sz), np.uint8); cv2.circle(m, (sz // 2, sz // 2), sz // 2, 1, -1)
                region = strip[y:y + sz, x:x + sz]
                region[m > 0] = obj[m > 0]
            else:
                strip[y:y + sz, x:x + sz] = obj
        else:  # a drawn decoration
            col = tuple(int(v) for v in rng.integers(0, 256, 3)) if out.ndim == 3 else int(rng.integers(0, 256))
            cx, cy, r = x + sz // 2, y + sz // 2, sz // 2
            k = rng.random()
            if k < 0.3:  # leaf
                cv2.ellipse(strip, (cx, cy), (max(1, r // 2), r), float(rng.uniform(-50, 50)), 0, 360, col, -1, cv2.LINE_AA)
            elif k < 0.6:  # flower
                for t in range(6):
                    a = np.pi * t / 3
                    cv2.circle(strip, (int(cx + 0.5 * r * np.cos(a)), int(cy + 0.5 * r * np.sin(a))), max(1, int(0.4 * r)), col, -1, cv2.LINE_AA)
            elif k < 0.8:  # star
                pts = [(cx + (r if t % 2 == 0 else 0.45 * r) * np.cos(np.pi / 2 + t * np.pi / 5),
                        cy - (r if t % 2 == 0 else 0.45 * r) * np.sin(np.pi / 2 + t * np.pi / 5)) for t in range(10)]
                cv2.fillPoly(strip, [np.array(pts, np.int32)], col, cv2.LINE_AA)
            else:  # heart
                cv2.circle(strip, (cx - r // 3, cy - r // 4), max(1, r // 2), col, -1, cv2.LINE_AA)
                cv2.circle(strip, (cx + r // 3, cy - r // 4), max(1, r // 2), col, -1, cv2.LINE_AA)
                cv2.fillPoly(strip, [np.array([(cx - r, cy - r // 6), (cx + r, cy - r // 6), (cx, cy + r)], np.int32)], col, cv2.LINE_AA)
        return np.concatenate([out, strip], axis=1) if rng.random() < 0.65 else np.concatenate([strip[:, ::-1], out], axis=1)

    MIN_GLYPH_PX = 18  # letter height as the model sees it (canvas 48 x 192, keep-ratio)

    def _big_enough(self, out):
        """Letters must stay readable after the keep-ratio fit to 48 x 192."""
        h, w = out.shape[:2]
        scale = min(48.0 / max(1, h), 192.0 / max(1, w))
        return getattr(self, "_last_glyph_h", h) * scale >= self.MIN_GLYPH_PX

    # ------------------------------------------------------------------ segment displays
    SEG7 = {"0": "ABCDEF", "1": "BC", "2": "ABGEDX", "3": "ABGXCD", "4": "FGXBC", "5": "AFGXCD",
            "6": "AFEDCGX", "7": "ABC", "8": "ABCDEFGX", "9": "ABCDFGX"}
    # 14-segment: A top, B upper right, C lower right, D bottom, E lower left, F upper left,
    # G / X middle left / right, H / J upper diagonals, I upper centre, K / M lower diagonals, L lower centre
    SEG14 = {"a": "ABCEFGX", "b": "ABCDILX", "c": "ADEF", "d": "ABCDIL", "e": "ADEFGX", "f": "AEFG",
             "g": "ACDEFX", "h": "BCEFGX", "i": "ADIL", "j": "BCDE", "k": "EFGJM", "l": "DEF",
             "m": "BCEFHJ", "n": "BCEFHM", "o": "ABCDEF", "p": "ABEFGX", "q": "ABCDEFM", "r": "ABEFGXM",
             "s": "AFGXCD", "t": "AIL", "u": "BCDEF", "v": "EFKJ", "w": "BCEFKM", "x": "HJKM",
             "y": "HJL", "z": "ADJK"}

    def _segment_word(self, rng):
        """A calculator / clock / LED display word: 7-segment digits or 14-segment letters."""
        if rng.random() < 0.6:  # numbers, sometimes with a decimal point or colon
            core = "".join(str(int(rng.integers(10))) for _ in range(int(rng.integers(1, 9))))
            seg14 = False
        else:
            n = int(rng.integers(2, 8))
            bucket = self.by_len.get(n) or self.lexicon or ["text"]
            core = bucket[int(rng.integers(len(bucket)))]
            seg14 = True
        ch_h = int(rng.integers(44, 80)); ch_w = int(ch_h * rng.uniform(0.5, 0.65))
        t = max(3, int(ch_h * rng.uniform(0.08, 0.15)))  # bar thickness
        gap = int(ch_w * rng.uniform(0.25, 0.45)); slant = rng.uniform(0, 0.2) if rng.random() < 0.6 else 0.0
        extras = {}
        if not seg14 and len(core) >= 2 and rng.random() < 0.35:  # decimal point / colon after a digit
            extras[int(rng.integers(0, len(core) - 1))] = "." if rng.random() < 0.7 else ":"
        W = int(len(core) * (ch_w + gap) + gap + slant * ch_h); H = int(ch_h * 1.35)
        style = rng.random()
        if style < 0.5:  # LED on black
            bg_c = rng.integers(0, 30, 3); on = [(40, 40, 255), (60, 255, 60), (255, 230, 60), (60, 200, 255), (255, 255, 255)][int(rng.integers(5))]
        elif style < 0.8:  # LCD: dark segments on grey-green
            bg_c = np.array([int(rng.integers(140, 200)), int(rng.integers(170, 220)), int(rng.integers(140, 200))]); on = tuple(int(v) for v in rng.integers(10, 50, 3))
        else:  # any colours
            bg_c = rng.integers(0, 256, 3); on = self._contrast_color(bg_c[::-1].astype(float), rng)
        img = np.empty((H, W, 3), np.uint8); img[:] = bg_c
        ghost = rng.random() < 0.5
        ghost_c = tuple(int(v) for v in (np.array(bg_c) * 0.9 + np.array(on) * 0.1))
        mask = np.zeros((H, W), np.uint8)
        y0 = int((H - ch_h) / 2)

        def segs(x0):
            h2 = ch_h / 2; cxm = x0 + ch_w / 2; d = t * 0.6
            return {"A": [(x0 + d, y0), (x0 + ch_w - d, y0)], "D": [(x0 + d, y0 + ch_h), (x0 + ch_w - d, y0 + ch_h)],
                    "G": [(x0 + d, y0 + h2), (cxm - d / 2, y0 + h2)], "X": [(cxm + d / 2, y0 + h2), (x0 + ch_w - d, y0 + h2)],
                    "F": [(x0, y0 + d), (x0, y0 + h2 - d)], "E": [(x0, y0 + h2 + d), (x0, y0 + ch_h - d)],
                    "B": [(x0 + ch_w, y0 + d), (x0 + ch_w, y0 + h2 - d)], "C": [(x0 + ch_w, y0 + h2 + d), (x0 + ch_w, y0 + ch_h - d)],
                    "I": [(cxm, y0 + d), (cxm, y0 + h2 - d)], "L": [(cxm, y0 + h2 + d), (cxm, y0 + ch_h - d)],
                    "H": [(x0 + d, y0 + d), (cxm - d / 2, y0 + h2 - d)], "J": [(x0 + ch_w - d, y0 + d), (cxm + d / 2, y0 + h2 - d)],
                    "K": [(cxm - d / 2, y0 + h2 + d), (x0 + d, y0 + ch_h - d)], "M": [(cxm + d / 2, y0 + h2 + d), (x0 + ch_w - d, y0 + ch_h - d)]}

        def sh(pt):  # italic slant: shift x by height above the baseline
            return (int(pt[0] + slant * (y0 + ch_h - pt[1])), int(pt[1]))

        table = self.SEG14 if seg14 else self.SEG7
        all_segs = "ABCDEFGXHIJKLM" if seg14 else "ABCDEFGX"
        for k, ch in enumerate(core):
            x0 = gap + k * (ch_w + gap)
            S_ = segs(x0)
            lit = table.get(ch, "")
            for name in all_segs:
                if name not in S_:
                    continue
                p0, p1 = sh(S_[name][0]), sh(S_[name][1])
                if name in lit:
                    cv2.line(img, p0, p1, on, t, cv2.LINE_AA); cv2.line(mask, p0, p1, 255, t, cv2.LINE_AA)
                elif ghost:
                    cv2.line(img, p0, p1, ghost_c, t, cv2.LINE_AA)
            if k in extras:
                px = int(x0 + ch_w + gap / 2 + slant * 0)
                if extras[k] == ".":
                    cv2.circle(img, sh((px, y0 + ch_h)), max(2, t // 1), on, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, sh((px, y0 + int(ch_h * 0.3))), max(2, t // 1), on, -1, cv2.LINE_AA)
                    cv2.circle(img, sh((px, y0 + int(ch_h * 0.7))), max(2, t // 1), on, -1, cv2.LINE_AA)
        if style < 0.5 and rng.random() < 0.6:  # LED glow
            img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(img, (0, 0), max(1.0, t * 0.8)), 0.7, 0)
        bg = np.empty_like(img); bg[:] = bg_c
        self._last_glyph_h = ch_h
        return img, bg, mask > 127, core

    # ------------------------------------------------------------------ logo-style arc
    def coco_images(self):
        if getattr(self, "_coco", None) is None:
            import pickle

            f = os.path.join(CACHE, "bg_coco.pkl")
            self._coco = pickle.load(open(f, "rb")) if os.path.isfile(f) else self.full_images()
        return self._coco

    def _real_blob(self, size, rng, gray=False):
        """A piece of a real COCO photo cut out with a random wiggly outline: (patch, mask)."""
        pool = self.coco_images()
        item = pool[int(rng.integers(len(pool)))]
        buf, boxes = item if isinstance(item, tuple) else (item, [])
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        H0, W0 = img.shape[:2]
        cs = max(8, int(min(H0, W0) * rng.uniform(0.12, 0.4)))
        for _ in range(8):  # avoid real text inside the piece
            y0, x0 = int(rng.integers(0, H0 - cs + 1)), int(rng.integers(0, W0 - cs + 1))
            if all(x0 + cs <= b[0] or x0 >= b[2] or y0 + cs <= b[1] or y0 >= b[3] for b in boxes):
                break
        patch = cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (size, size), interpolation=cv2.INTER_AREA)
        # wiggly outline: random radius per angle, smoothed, as a filled polygon
        n = 48
        rad = rng.uniform(0.55, 1.0, int(rng.integers(5, 10)))
        rad = np.interp(np.linspace(0, len(rad), n, endpoint=False), np.arange(len(rad) + 1), np.r_[rad, rad[0]])
        rad = cv2.GaussianBlur(rad[None].astype(np.float32), (1, 7), 0)[0] if n >= 7 else rad
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = size / 2
        pts = np.stack([c + rad * c * 0.95 * np.cos(ang), c + rad * c * 0.95 * np.sin(ang)], 1).astype(np.int32)
        mask = np.zeros((size, size), np.uint8)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
        if gray:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return patch, mask

    def _paste_blob(self, dst, cx, cy, size, rng):
        if not self.coco_images():
            return False
        patch, mask = self._real_blob(size, rng, gray=dst.ndim == 2)
        x0, y0 = int(cx - size / 2), int(cy - size / 2)
        x1, y1 = x0 + size, y0 + size
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(dst.shape[1], x1), min(dst.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return False
        p = patch[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
        a = (mask[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32) / 255.0)
        a = a[..., None] if dst.ndim == 3 else a
        region = dst[y0:y1, x0:x1].astype(np.float32)
        dst[y0:y1, x0:x1] = (region * (1 - a) + p * a).astype(np.uint8)
        return True

    def _decoration(self, img, cx, cy, size, rng):
        if self._paste_blob(img, cx, cy, size, rng):  # always a real COCO piece (no drawn shapes)
            return
        """Flower / tree / leaves / star / sun drawn at (cx, cy) with the given size (BGR)."""
        r = max(3, size // 2)
        green = (int(rng.integers(20, 90)), int(rng.integers(120, 220)), int(rng.integers(20, 90)))
        bright = tuple(int(v) for v in rng.integers(150, 256, 3))
        kind = rng.random()
        if kind < 0.3:  # tree: trunk + round canopy (or triangle)
            cv2.rectangle(img, (cx - max(1, r // 6), cy), (cx + max(1, r // 6), cy + r), (30, 60, 110), -1)
            if rng.random() < 0.5:
                cv2.circle(img, (cx, cy - r // 4), int(0.75 * r), green, -1, cv2.LINE_AA)
            else:
                cv2.fillPoly(img, [np.array([(cx - r, cy + r // 3), (cx + r, cy + r // 3), (cx, cy - r)], np.int32)], green, cv2.LINE_AA)
        elif kind < 0.55:  # flower
            n = int(rng.integers(5, 9)); pc = bright if rng.random() < 0.6 else (255, 255, 255)
            for k in range(n):
                a = 2 * np.pi * k / n
                cv2.circle(img, (int(cx + 0.5 * r * np.cos(a)), int(cy + 0.5 * r * np.sin(a))), max(2, int(0.36 * r)), pc, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), max(2, int(0.25 * r)), (40, 200, 240), -1, cv2.LINE_AA)
        elif kind < 0.75:  # leaves / branch
            for k in range(int(rng.integers(2, 5))):
                ang = float(rng.uniform(-70, 70)); off = int((k - 1) * 0.4 * r)
                cv2.ellipse(img, (cx + off, cy), (max(2, int(0.25 * r)), max(3, int(0.6 * r))), ang, 0, 360, green, -1, cv2.LINE_AA)
        elif kind < 0.9:  # star
            pts = [(cx + (r if k % 2 == 0 else 0.45 * r) * np.cos(np.pi / 2 + k * np.pi / 5),
                    cy - (r if k % 2 == 0 else 0.45 * r) * np.sin(np.pi / 2 + k * np.pi / 5)) for k in range(10)]
            cv2.fillPoly(img, [np.array(pts, np.int32)], bright, cv2.LINE_AA)
        else:  # sun
            cv2.circle(img, (cx, cy), int(0.5 * r), bright, -1, cv2.LINE_AA)
            for k in range(8):
                a = np.pi * k / 4
                cv2.line(img, (int(cx + 0.65 * r * np.cos(a)), int(cy + 0.65 * r * np.sin(a))),
                         (int(cx + r * np.cos(a)), int(cy + r * np.sin(a))), bright, max(1, r // 8), cv2.LINE_AA)

    def _logo_arc(self, rng):
        """Round-sign / logo style: big bold light letters on a dark background, written along a
        centred arc (rainbow or smile, 70-140 deg), with a decoration inside the curve."""
        if self.lexicon and rng.random() < 0.8:
            bucket = [w for w in (self.by_len.get(int(rng.integers(3, 11))) or self.lexicon)]
            core = bucket[int(rng.integers(len(bucket)))]
        else:
            core = "".join(str(int(rng.integers(10))) for _ in range(int(rng.integers(2, 6))))
        shown = core.upper() if rng.random() < 0.7 else core.title()
        bold = [f for f in self.fonts if re.search(r"bold|black|heavy|extrab", os.path.basename(f), re.I)] or self.fonts
        pool = self.fancy if (self.fancy and rng.random() < 0.15) else bold
        font = ImageFont.truetype(pool[int(rng.integers(len(pool)))], int(rng.integers(46, 76)))
        stroke = int(rng.choice([0, 0, 1, 2]))
        asc, desc = font.getmetrics()
        adv = [max(2.0, font.getlength(ch)) + stroke + 2 for ch in shown]
        L = float(sum(adv))
        span = np.radians(rng.uniform(70, 140))
        R = max(L / span, 1.8 * (asc + desc)); span = L / R
        below = rng.random() < 0.35
        S = int(2.2 * (asc + desc) + 2 * stroke + 4)
        placed, cum = [], 0.0
        for ch, a in zip(shown, adv):
            th = -span / 2 + (cum + a / 2) / R; cum += a
            if not below:
                placed.append((ch, R * np.sin(th), -R * np.cos(th), -np.degrees(th)))
            else:
                placed.append((ch, R * np.sin(th), R * np.cos(th), np.degrees(th)))
        xs = [p[1] for p in placed]; ys = [p[2] for p in placed]
        m = S // 2 + 6
        # decoration inside the curve: towards the circle centre (0, 0)
        dec_size = int(min(0.8 * R, 1.4 * (asc + desc)) * rng.uniform(0.6, 1.0))
        dec_y = (-R + asc + desc + dec_size * 0.6) if not below else (R - asc - desc - dec_size * 0.6)
        x_min = min(xs) - m; y_min = min(min(ys) - m, dec_y - dec_size)
        W = int(max(xs) - x_min + m); H = int(max(max(ys) + m, dec_y + dec_size) - y_min)
        dark = rng.random() < 0.8
        base = rng.integers(0, 45, 3) if dark else rng.integers(20, 120, 3)
        bg = np.empty((H, W, 3), np.uint8); bg[:] = base
        wheel = rng.random() < 0.4  # half-wheel / tyre look: dark disc on a lighter background
        if wheel:
            bg[:] = rng.integers(140, 256, 3)
            cx_, cy_ = int(-x_min), int(-y_min)
            r_out = int(R + asc + desc + rng.uniform(4, 14))
            cv2.circle(bg, (cx_, cy_), r_out, tuple(int(v) for v in base), -1, cv2.LINE_AA)
            if rng.random() < 0.5:  # hub: lighter inner circle
                cv2.circle(bg, (cx_, cy_), max(4, int(R - (asc + desc) * 0.4)), tuple(int(v) for v in np.clip(base + rng.integers(20, 60), 0, 255)), -1, cv2.LINE_AA)
        if rng.random() < 0.3:  # faint texture
            bg = np.clip(bg + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)
        col = tuple(int(v) for v in (rng.integers(225, 256, 3) if rng.random() < 0.7 else rng.integers(150, 256, 3)))
        canvas = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).convert("RGBA")
        mask = Image.new("L", (W, H), 0)
        for ch, px, py, rot in placed:
            tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((S / 2 - font.getlength(ch) / 2, S / 2 - asc), ch, font=font, fill=col + (255,), stroke_width=stroke)
            tile = tile.rotate(rot, resample=Image.BICUBIC, center=(S / 2, S / 2))
            pos = (int(round(px - x_min - S / 2)), int(round(py - y_min - S / 2)))
            canvas.alpha_composite(tile, dest=(max(0, pos[0]), max(0, pos[1])), source=(max(0, -pos[0]), max(0, -pos[1])))
            mask.paste(tile.split()[3], pos, tile.split()[3])
        out = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        mk = np.array(mask) > 127
        # decoration (and sometimes a second one) inside the curve, never on the letters
        cx0 = int(-x_min)
        for k in range(1 if rng.random() < 0.6 else 2):
            dx = 0 if k == 0 else int(rng.uniform(-0.35, 0.35) * R)
            dec = out.copy()
            self._decoration(dec, cx0 + dx, int(dec_y - y_min), dec_size if k == 0 else int(dec_size * 0.6), rng)
            changed = np.abs(dec.astype(np.int16) - out.astype(np.int16)).max(2) > 0
            if not (changed & cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).any():
                out = dec
        ys_, xs_ = np.where(mk | (np.abs(out.astype(np.int16) - bg.astype(np.int16)).max(2) > 0))
        pad = 6
        y0, y1 = max(0, ys_.min() - pad), min(H, ys_.max() + pad); x0, x1 = max(0, xs_.min() - pad), min(W, xs_.max() + pad)
        self._last_glyph_h = 0.75 * (asc + desc)
        return out[y0:y1, x0:x1], bg[y0:y1, x0:x1], mk[y0:y1, x0:x1], core.lower()

    def _plaque_rim(self, rng):
        """House-name plaque: light serif capitals along the rim of a dark round/oval sign, with a
        thin outer ring, a light wall outside and a painted scene inside. The crop is an
        OFF-CENTRE slice of the rim, so the letters lean one way and the end letters reach 40-75 deg,
        and the word is cropped tight to the image edges (IIIT5K honeybee / point / farthings / croft)."""
        if self.lexicon and rng.random() < 0.85:
            bucket = self.by_len.get(int(rng.integers(3, 11))) or self.lexicon
            core = bucket[int(rng.integers(len(bucket)))]
        else:
            core = "".join(str(int(rng.integers(10))) for _ in range(int(rng.integers(2, 5))))
        shown = core.upper() if rng.random() < 0.85 else core.title()
        sp = getattr(self, "_serif", None)
        if sp is None:
            pat = re.compile(r"serif|roman|times|c059|p052|garamond|baskerv|cinzel|trajan|playfair|lora|merriweather|crimson|bodoni|caslon|cormorant|spectral|tinos|libre.?baskerville|marcellus|cardo|prata|frank.?ruhl|old.?standard|gelasio|domine|vollkorn|alegreya$|rosarivo|forum|abhaya|eb.?garamond", re.I)
            sp = self._serif = [f for f in self.fonts + self.fancy if pat.search(os.path.basename(f)) and not re.search(r"sans|italic|oblique|script|hand", f, re.I)] or self.fonts
        font = ImageFont.truetype(sp[int(rng.integers(len(sp)))], int(rng.integers(44, 72)))
        stroke = int(rng.choice([0, 0, 0, 1]))
        asc, desc = font.getmetrics(); ch_h = asc + desc
        adv = [max(2.0, font.getlength(ch)) * rng.uniform(1.0, 1.25) + stroke + 3 for ch in shown]
        L = float(sum(adv))
        span = np.radians(rng.uniform(70, 140))
        R = max(L / span, 1.9 * ch_h); span = L / R
        # off-centre: the word's middle sits 0-50 deg from the top, end letters up to ~75 deg
        u = rng.random()  # 60% diagonal slice (one side of the circle), 25% mildly off-centre, 15% centred
        mid = np.radians(rng.choice([-1, 1]) * rng.uniform(35, 65)) if u < 0.6 else np.radians(rng.uniform(-30, 30)) if u < 0.85 else 0.0
        mid = float(np.clip(mid, -np.radians(82) + span / 2, np.radians(82) - span / 2)) if span / 2 < np.radians(82) else 0.0
        below = rng.random() < getattr(self, "p_plaque_below", 0.25)
        S = int(2.2 * ch_h + 2 * stroke + 4)
        placed, cum = [], 0.0
        for ch, a in zip(shown, adv):
            th = mid - span / 2 + (cum + a / 2) / R; cum += a
            if not below:
                placed.append((ch, R * np.sin(th), -R * np.cos(th), -np.degrees(th)))
            else:
                placed.append((ch, R * np.sin(th), R * np.cos(th), np.degrees(th)))
        # canvas centred on the circle centre
        r_disc = R + 0.62 * ch_h + rng.uniform(2, 10)   # dark band ends just outside the letters
        r_ring = r_disc + rng.uniform(2, 7)              # thin outer ring
        Wc = Hc = int(2 * (r_ring + ch_h)); c0 = Wc // 2
        wall = rng.integers(170, 250); bg = np.empty((Hc, Wc, 3), np.uint8)
        bg[:] = np.clip(wall + rng.integers(-15, 15, 3), 0, 255)
        if rng.random() < 0.5:  # soft lighting gradient on the wall
            gx = np.linspace(-1, 1, Wc)[None, :, None] * rng.uniform(-25, 25)
            bg = np.clip(bg + gx, 0, 255).astype(np.uint8)
        dark = tuple(int(v) for v in rng.integers(0, 50, 3))
        ring = tuple(int(v) for v in np.clip(np.array(dark) + rng.integers(-10, 40, 3), 0, 255))
        cv2.circle(bg, (c0, c0), int(r_ring), ring, -1, cv2.LINE_AA)
        if rng.random() < 0.6:  # light gap between ring and disc
            cv2.circle(bg, (c0, c0), int((r_ring + r_disc) / 2), tuple(int(v) for v in rng.integers(90, 200, 3)), max(1, int(rng.uniform(1, 3))), cv2.LINE_AA)
        cv2.circle(bg, (c0, c0), int(r_disc), dark, -1, cv2.LINE_AA)
        if rng.random() < 0.4:
            bg = np.clip(bg + rng.normal(0, 5, bg.shape), 0, 255).astype(np.uint8)
        # painted scene inside the circle of letters
        r_in = int(R - 0.62 * ch_h - rng.uniform(3, 12))
        scene = bg.copy()
        if r_in > 20 and rng.random() < 0.7:
            col = tuple(int(v) for v in rng.integers(60, 240, 3))
            cv2.circle(scene, (c0, c0), r_in, col, -1, cv2.LINE_AA)
            if rng.random() < 0.6:  # ground band
                band = np.zeros(scene.shape[:2], np.uint8); cv2.circle(band, (c0, c0), r_in, 255, -1)
                yb = c0 + int(rng.uniform(-0.2, 0.5) * r_in) if not below else c0 - int(rng.uniform(-0.2, 0.5) * r_in)
                sl = slice(yb, None) if not below else slice(None, yb)
                m2 = np.zeros_like(band); m2[sl] = 255; m2 &= band
                scene[m2 > 0] = rng.integers(30, 160, 3)
        yy, xx = np.mgrid[0:Hc, 0:Wc]; dist = np.hypot(xx - c0, yy - c0)
        if self.coco_images():
            if r_in > 20 and rng.random() < 0.5:  # a real photo crop fills the inner circle (painted scene)
                buf = self.coco_images()[int(rng.integers(len(self.coco_images())))]
                ph = cv2.imdecode(np.frombuffer(buf[0] if isinstance(buf, tuple) else buf, np.uint8), cv2.IMREAD_COLOR)
                if ph is not None:
                    h0, w0 = ph.shape[:2]; cs = int(min(h0, w0) * rng.uniform(0.4, 1.0))
                    py0, px0 = int(rng.integers(0, h0 - cs + 1)), int(rng.integers(0, w0 - cs + 1))
                    ph = cv2.resize(ph[py0:py0 + cs, px0:px0 + cs], (2 * r_in, 2 * r_in))
                    inner = dist[c0 - r_in:c0 + r_in, c0 - r_in:c0 + r_in] < r_in
                    scene[c0 - r_in:c0 + r_in, c0 - r_in:c0 + r_in][inner] = ph[inner]
            for _ in range(int(rng.integers(0, 4))):  # wiggly photo pieces inside the circle
                if r_in < 25: break
                sz = int(rng.uniform(0.3, 0.8) * r_in); a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0, max(1, r_in - sz / 2))
                tmp = scene.copy(); self._paste_blob(tmp, int(c0 + rr * np.cos(a)), int(c0 + rr * np.sin(a)), sz, rng)
                keep = dist < r_in; scene[keep] = tmp[keep]
            for _ in range(int(rng.integers(0, 4)) if rng.random() < 0.6 else 0):  # photo pieces on the wall, outside the ring
                sz = int(rng.uniform(0.6, 2.0) * ch_h); a = rng.uniform(0, 2 * np.pi); rr = r_ring + rng.uniform(0.2, 1.0) * ch_h
                tmp = scene.copy(); self._paste_blob(tmp, int(c0 + rr * np.cos(a)), int(c0 + rr * np.sin(a)), sz, rng)
                keep = dist > r_ring + 1; scene[keep] = tmp[keep]
        tcol = tuple(int(v) for v in (rng.integers(225, 256, 3) if rng.random() < 0.8 else rng.integers(170, 256, 3)))
        canvas = Image.fromarray(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)).convert("RGBA")
        mask = Image.new("L", (Wc, Hc), 0)
        for ch, px, py, rot in placed:
            tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((S / 2 - font.getlength(ch) / 2, S / 2 - asc), ch, font=font, fill=tcol + (255,), stroke_width=stroke)
            tile = tile.rotate(rot, resample=Image.BICUBIC, center=(S / 2, S / 2))
            pos = (int(round(c0 + px - S / 2)), int(round(c0 + py - S / 2)))
            canvas.alpha_composite(tile, dest=(max(0, pos[0]), max(0, pos[1])), source=(max(0, -pos[0]), max(0, -pos[1])))
            mask.paste(tile.split()[3], pos, tile.split()[3])
        out = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
        mk = np.array(mask) > 127
        # decorations inside (never touching letters)
        for _ in range(int(rng.integers(0, 3))):
            if r_in < 25: break
            a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0, 0.7) * r_in
            dec = out.copy(); self._decoration(dec, int(c0 + rr * np.cos(a)), int(c0 + rr * np.sin(a)), int(rng.uniform(0.3, 0.7) * min(r_in, 1.5 * ch_h)), rng)
            changed = np.abs(dec.astype(np.int16) - out.astype(np.int16)).max(2) > 0
            if not (changed & cv2.dilate(mk.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)).any():
                out = dec
        # crop: tight around the letters (sometimes touching the border), more room towards the centre
        ys_, xs_ = np.where(mk)
        y0, y1, x0, x1 = ys_.min(), ys_.max(), xs_.min(), xs_.max()
        tight = rng.random() < 0.4
        e = lambda lo, hi: int(rng.uniform(lo, hi) * ch_h)
        pl, pr = (e(0.1, 0.18), e(0.1, 0.18)) if tight else (e(0.15, 0.45), e(0.15, 0.45))  # margin survives the _finish warp
        po, pin = e(0.15, 0.6), e(0.15, 1.3)   # outward (rim, wall) / inward (scene)
        if not below: y0, y1 = y0 - po, y1 + pin
        else: y0, y1 = y0 - pin, y1 + po
        x0, x1 = x0 - pl, x1 + pr
        y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(Hc, y1), min(Wc, x1)
        self._last_glyph_h = 0.75 * ch_h
        return out[y0:y1, x0:x1], bg[y0:y1, x0:x1], mk[y0:y1, x0:x1], core.lower()

    def _render(self, rng):
        if rng.random() < getattr(self, "p_segment", 0.0):
            out, bg, mk, label = self._segment_word(rng)
            self.last_arc = False
            return self._finish(out, bg, mk, label, rng)
        if rng.random() < getattr(self, "p_plaque", 0.0):
            out, bg, mk, label = self._plaque_rim(rng)
            self.last_arc = True
            res = self._finish(out, bg, mk, label, rng)
            self._last_glyph_h = getattr(self, "_last_glyph_h", 20)
            return res
        if rng.random() < getattr(self, "p_logo_arc", 0.0):
            out, bg, mk, label = self._logo_arc(rng)
            self.last_arc = True
            res = self._finish(out, bg, mk, label, rng)
            self._last_glyph_h = getattr(self, "_last_glyph_h", 20)
            return res
        shown, label = self.text(rng)
        arc = rng.random() < getattr(self, "p_arc_text", 0.0) and len(shown) >= 2
        self.last_arc = arc
        fancy = bool(self.fancy) and rng.random() < (0.35 if arc else self.p_fancy)
        pool = self.fancy if fancy else self.fonts
        if fancy and rng.random() < 0.25:
            sq = getattr(self, "_square", None)
            if sq is None:
                pat = re.compile(r"orbitron|silkscreen|pressstart|pixelify|geistpixel|micro5|jersey(10|15|20|25)$|novasquare|iceland|wallpoet|turret|oxanium|audiowide|gruppo|sairastencil", re.I)
                sq = self._square = [f for f in self.fancy if pat.search(os.path.basename(os.path.dirname(f)))]
            if sq:
                pool = sq
        font_path = pool[int(rng.integers(len(pool)))]
        size = int(rng.integers(28, 72))
        stroke = int(rng.choice([0, 0, 1, 2, 3, 4, 5]))  # thick fonts via stroke width
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.truetype(self.fonts[0], size)
        text_mask = None
        asc_, desc_ = font.getmetrics()
        self._last_glyph_h = 0.75 * (asc_ + desc_)  # approx. cap / x-height span of a letter
        if arc:
            out, bg, text_mask = self._arc_text(rng, shown, font, size, stroke)
            H, W = out.shape[:2]
            style = 1.0  # plain / per-char styles only
            return self._finish(out, bg, text_mask, label, rng)
        # measure
        tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        try:
            x0, y0, x1, y1 = tmp.textbbox((0, 0), shown, font=font, stroke_width=stroke)
        except Exception:
            x0, y0, x1, y1 = 0, 0, size * len(shown), size
        tw, th = max(8, x1 - x0), max(8, y1 - y0)
        pad_x, pad_y = int(rng.uniform(0.02, 0.25) * th) + stroke + 2, int(rng.uniform(0.05, 0.35) * th) + stroke + 2
        W, H = tw + 2 * pad_x, th + 2 * pad_y
        sign = rng.random() < 0.35 and not getattr(self, "_simple", False)
        if sign:
            bg, color = self._sign_pair(H, W, rng); self._last_bg_kind = "clean"
        else:
            bg = self.background(W, H, rng)
        im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(im)
        bg_mean = bg.reshape(-1, 3).mean(0)
        if not sign:
            color = self._contrast_color(bg_mean, rng)
        ox, oy = pad_x - x0, pad_y - y0
        if rng.random() < 0.35:  # shadow or outline
            off = int(rng.integers(1, max(2, size // 10)))
            sc = tuple(int(v) for v in rng.integers(0, 256, 3))
            if rng.random() < 0.5:
                draw.text((ox + off, oy + off), shown, font=font, fill=sc, stroke_width=stroke)
            else:
                draw.text((ox, oy), shown, font=font, fill=sc, stroke_width=stroke + off)
        style = rng.random()
        if style < 0.12:  # outline-only lettering
            sw = max(1, int(rng.integers(1, max(2, size // 12))))
            draw.text((ox, oy), shown, font=font, fill=tuple(int(v) for v in bg_mean[::-1]),
                      stroke_width=sw, stroke_fill=color)
            per_color = per_thick = False
        else:
            per_color, per_thick = rng.random() < 0.5, rng.random() < 0.3
        if style < 0.12:
            pass
        elif per_color or per_thick:  # per-character colours and / or stroke thickness
            x = ox
            for ch in shown:
                c = self._contrast_color(bg_mean, rng) if (per_color and rng.random() < 0.7) else color
                sw = int(rng.choice([0, 1, 2, 3, 4, 5])) if per_thick else stroke
                draw.text((x, oy), ch, font=font, fill=c, stroke_width=sw)
                try:
                    x += font.getlength(ch)
                except Exception:
                    x += size * 0.6
        else:
            draw.text((ox, oy), shown, font=font, fill=color, stroke_width=stroke)
        out = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        if 0.12 <= style < 0.2:  # LED / dot-matrix look: keep the text only on a dot grid
            mask = Image.new("L", (W, H), 0)
            ImageDraw.Draw(mask).text((ox, oy), shown, font=font, fill=255, stroke_width=stroke)
            m = np.array(mask) > 127
            step = max(2, int(size / rng.uniform(9, 15)))
            r = max(1, int(step * rng.uniform(0.3, 0.45)))
            dots = np.zeros((H, W), np.uint8)
            for yy in range(step // 2, H, step):
                for xx in range(step // 2, W, step):
                    if m[yy, xx]:
                        cv2.circle(dots, (xx, yy), r, 255, -1, cv2.LINE_AA)
            glow = np.array(color[::-1], np.float32)[None, None]
            a = (dots.astype(np.float32) / 255.0)[..., None]
            out = (bg.astype(np.float32) * (1 - a) + glow * a).astype(np.uint8)
            if rng.random() < 0.5:
                out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (0, 0), r), 0.6, 0)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).text((ox, oy), shown, font=font, fill=255, stroke_width=stroke)
        return self._finish(out, bg, np.array(mask) > 127, label, rng)

    def _finish(self, out, bg, mk, label, rng):
        """Blending, lighting, low-res look and the final legibility check (straight and arc text)."""
        H, W = out.shape[:2]
        blend_p = 0.5 if getattr(self, "_last_bg_kind", "") == "full" else 0.3
        if getattr(self, "_simple", False):
            blend_p = 0.0
        painted = out.copy()
        if rng.random() < blend_p:  # blend the text into the background instead of painting it
            diff = np.abs(out.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
            m = (diff > 8).astype(np.float32)
            hard = m > 0.5
            m = cv2.GaussianBlur(m, (0, 0), rng.uniform(0.4, 1.2))[..., None]
            mode = rng.random()
            if mode < 0.35:  # translucent paint
                a = rng.uniform(0.5, 0.85)
                out = (bg * (1 - a * m) + out * (a * m)).astype(np.uint8)
            elif mode < 0.75:  # darken / lighten the background under the strokes (keeps texture)
                k = rng.uniform(0.35, 0.7) if bg.mean() > 110 else rng.uniform(1.5, 2.6)
                if rng.random() < 0.3:
                    k = 1.0 / k  # the other direction
                tinted = bg.astype(np.float32) * k + (rng.integers(-25, 26, 3) if rng.random() < 0.5 else 0)
                out = np.clip(bg * (1 - m) + tinted * m, 0, 255).astype(np.uint8)
            else:  # noisy / worn stroke mask
                noise = cv2.GaussianBlur(rng.random(m.shape[:2]).astype(np.float32), (0, 0), rng.uniform(0.6, 2.0))
                mm = m * (noise[..., None] > rng.uniform(0.3, 0.5))
                a = rng.uniform(0.6, 0.95)
                out = (bg * (1 - a * mm) + out * (a * mm)).astype(np.uint8)
            # legibility check: text vs. its local surroundings (dilated ring around the strokes)
            if hard.any():
                g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
                ring = (cv2.dilate(hard.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~hard
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                cdiff = np.linalg.norm(lab[hard].mean(0) - lab[ring].mean(0)) if ring.any() else 0.0
                if cdiff < 28:  # too faint: stronger translucent blend, else solid paint
                    out = (bg * (1 - 0.9 * m) + painted * (0.9 * m)).astype(np.uint8)
                    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                    if ring.any() and np.linalg.norm(lab[hard].mean(0) - lab[ring].mean(0)) < 28:
                        out = painted
        # illumination: vignette / directional light
        if rng.random() < 0.4:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(W, H)
            light = (1.0 - rng.uniform(0.1, 0.4) * d)[..., None]
            out = np.clip(out * light + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        if rng.random() < 0.08:  # low-resolution look (mild; the augmentation adds more)
            f = rng.uniform(0.6, 0.85)
            out = cv2.resize(out, (max(8, int(W * f)), max(6, int(H * f))), interpolation=cv2.INTER_AREA)
            self._last_glyph_h = getattr(self, "_last_glyph_h", H) * f
        # legibility of the final image: text pixels vs. a ring around them
        if out.shape[:2] != (H, W):
            mk = cv2.resize(mk.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ok = True
        if mk.sum() > 10:
            ring = (cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~mk
            if ring.sum() > 10:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                d = float(np.linalg.norm(lab[mk].mean(0) - lab[ring].mean(0)))
                clutter = 0.5 * (float(lab[ring].std(0).mean()) + float(lab[mk].std(0).mean()))
                # mean difference large AND large relative to the clutter around the strokes
                ok = d >= self.MIN_CONTRAST and d >= 3.0 * clutter
                if ok:  # every part of the word must stand out (per-character colours / scenes)
                    xs = np.where(mk.any(0))[0]
                    nseg = max(1, min(len(label) if label else 1, 12))
                    edges = np.linspace(xs.min(), xs.max() + 1, nseg + 1).astype(int)
                    for a_, b_ in zip(edges[:-1], edges[1:]):
                        seg = np.zeros_like(mk); seg[:, a_:b_] = True
                        sm, sr = mk & seg, ring & seg
                        if sm.sum() > 6 and sr.sum() > 6:
                            if float(np.linalg.norm(lab[sm].mean(0) - lab[sr].mean(0))) < 0.8 * self.MIN_CONTRAST:
                                ok = False
                                break
        if not self.color:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return out, label, ok


def load_lexicon():
    path = os.path.join(CACHE, "lexicon.txt")
    if os.path.isfile(path):
        return [l.strip() for l in open(path) if l.strip()]
    return []


if __name__ == "__main__":  # quick sheet
    r = Renderer(load_lexicon(), color=True)
    print(len(r.fonts), "fonts")
    tiles = []
    for i in range(24):
        img, lab = r.render(np.random.default_rng(i))
        t = cv2.resize(img, (220, 56))
        t = cv2.copyMakeBorder(t, 0, 0, 0, 110, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, lab, (224, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(t)
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, 24, 3)]
    cv2.imwrite(os.path.join(CACHE, "render_sheet.png"), np.concatenate(rows, 0))
    print("sheet -> data_cache/render_sheet.png")
