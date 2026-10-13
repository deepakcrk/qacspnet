#!/usr/bin/env python3
# Train / evaluate QA-CSPTextNet (scene text recognition; CTC or attention decoder).
#
# Data sources (--train / --val / --test), any mix of:
#   * label file: one sample per line, "<image path> <word>" (paths relative to --root)
#   * LMDB folder (data.mdb with image-%09d / label-%09d / num-samples)
#   * folder of LMDB folders (e.g. MJSynth chunk_00 ... chunk_19)
# --alphabet sets the character set (default 0-9a-z). Labels are lower-cased unless
# the alphabet contains upper-case letters. Training samples with characters outside
# the alphabet, empty labels or more than --max-len characters are skipped (CTC has
# 42 time steps at the default 32x96 input; --img-size H W changes it). Evaluation keeps every sample; characters outside the alphabet are
# dropped from the ground truth before comparison.
#   python tools/train_textnet.py --variant s --root data/ \
#       --train mjsynth_train.txt --val iiit5k_test.txt svt_test.txt
#   python tools/train_textnet.py --variant s --root data/ --val iiit5k_test.txt \
#       --eval-only --ckpt textnet_s/best.pth
#
# Model selection uses --val (a dev split, e.g. held-out synthetic or IIIT5K train);
# --test files are evaluated once with the selected checkpoint.
# Repeated runs: --seed N --metrics-out textnet_s_seedN.json

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from qacspnet.models import QACSPTextNet

VARIANTS = {"s": (0.33, 0.5), "m": (0.67, 0.5), "l": (1.0, 0.5), "x": (1.33, 0.5)}


def _chw(img):
    """HxW (gray) or HxWx3 (BGR) uint8 -> float tensor C x H x W in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0).sub_(0.5).div_(0.5)
    return t[None] if t.ndim == 2 else t.permute(2, 0, 1)


def _to_tensor(img, img_size):
    img = cv2.resize(img, (img_size[1], img_size[0]))
    return _chw(img)


def fit_keep_ratio(img, img_size):
    """Aspect-preserving resize to height H, left-aligned on an H x W canvas padded with the
    median of the last column (wider words are squeezed to W). Same as exploreSTR-wide."""
    h, w = img_size
    ih, iw = img.shape[:2]
    nw = max(1, min(w, int(round(iw * h / ih))))
    g = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA if ih > h else cv2.INTER_CUBIC)
    if g.ndim == 3:
        out = np.empty((h, w, g.shape[2]), np.uint8)
        out[:] = np.median(g[:, -1], axis=0).astype(np.uint8)
    else:
        out = np.full((h, w), int(np.median(g[:, -1])), np.uint8)
    out[:, :nw] = g
    return out


def prepare(ds, img, i):
    """Raw grayscale crop -> model input tensor, following the dataset's preprocessing."""
    tf = getattr(ds, "transform", None)
    pre = getattr(ds, "pre_transform", None)  # augmentation on the raw crop (bend / symbols)
    if pre is not None and img is not None and img.ndim >= 2 and img.shape[0] >= 4 and img.shape[1] >= 4:
        img = pre(img, i, ds)
    if getattr(ds, "keep_ratio", False):
        img = fit_keep_ratio(img, ds.img_size)
        if tf is not None:
            img = tf(img, i)  # augmentation on the fixed-size canvas
        return _chw(img)
    if tf is not None:
        img = tf(img, i)
    return _to_tensor(img, ds.img_size)


class CurveSymbolAugment:
    """Applied to the raw crop (before the resize). Two independent steps:
    bend: warp the text along an arc (parabola, up or down) or a mild sine wave, so the
          model learns curved text;
    symbols: draw punctuation / symbols (- . , ! ' : _ * " / |) at the start, end, between the
          characters or above the text, in a contrasting colour. Labels are letters and digits
          only, so the model learns to ignore such marks."""

    SYMS = (list("-.,!':_*\"/|\\()[]?&+=#@%~;") + ["...", "--", "..", "!!", "**", "//", "\\\\", "/ /"]
            + list("{}<>") + ["()", "[]", "{}", "( )", "[ ]"]
            + ["("] * 2 + [")"] * 2 + ["["] * 2 + ["]"] * 2 + ["{"] * 2 + ["}"] * 2
            + ["/"] * 4 + ["\\"] * 4 + ["-"] * 3 + ["."] * 3 + ["!"] * 2
            + ["$"] * 3 + ["\""] * 2 + ["'"] * 2 + [","] * 2 + [":"] * 2 + ["?"] * 2 + ["\"\""] + ["''"]
            + ["$ "] + ["(" ] + [")"])  # also $, quotes, apostrophe, comma, colon, question mark

    def __init__(self, p_bend=0.5, p_symbols=0.5, max_bend=0.35, p_blur=0.0, p_erase=0.0,
                 p_edge=0.0, p_lines=0.0, p_geom=0.0, p_tone=0.0):
        # readability caps (user, 2026-10-11): every training image must stay readable by a person
        # geometry (curve / rotation / slant) is emphasised; degradations are kept low
        p_bend, p_symbols, p_blur = min(max(p_bend, 0.45), 0.45), min(p_symbols, 0.25), min(p_blur, 0.1)
        p_erase, p_lines, p_geom = min(p_erase, 0.1), min(p_lines, 0.08), min(max(p_geom, 0.35), 0.35)
        self.p_bend, self.p_symbols, self.max_bend = p_bend, p_symbols, max_bend
        self.p_blur, self.p_erase, self.p_edge, self.p_lines = p_blur, p_erase, p_edge, p_lines
        self.p_geom, self.p_tone = p_geom, p_tone
        self.max_span = 110

    def tone(self, img, rng):
        """Each character (approximated by vertical segments) gets its own brightness /
        contrast, and in colour mode its own tint, with smooth boundaries."""
        h, w = img.shape[:2]
        n = int(rng.integers(2, 9))
        cuts = np.sort(rng.integers(1, max(2, w), n - 1)) if n > 1 else np.array([], int)
        seg = np.zeros(w, np.int32)
        for c in cuts: seg[c:] += 1
        gains = rng.uniform(0.55, 1.45, n); offs = rng.uniform(-50, 50, n)
        gmap = gains[seg].astype(np.float32); omap = offs[seg].astype(np.float32)
        k = max(1, int(rng.uniform(0.02, 0.12) * w)) * 2 + 1
        gmap = cv2.GaussianBlur(gmap[None, :], (k, 1), 0)[0]; omap = cv2.GaussianBlur(omap[None, :], (k, 1), 0)[0]
        out = img.astype(np.float32)
        if out.ndim == 3:
            tint = rng.uniform(0.7, 1.3, (n, 3))[seg].astype(np.float32)
            tint = cv2.GaussianBlur(tint[None], (k, 1), 0)[0]
            out = out * (gmap[None, :, None] * tint[None]) + omap[None, :, None]
        else:
            out = out * gmap[None, :] + omap[None, :]
        return np.clip(out, 0, 255).astype(np.uint8)

    def geom(self, img, rng):
        """Strong slant (shear), rotation and a small perspective tilt; the canvas is enlarged
        so that no character is cut off."""
        h, w = img.shape[:2]
        ang = rng.uniform(-12, 12)
        sh = rng.uniform(-0.35, 0.35)
        pad_x, pad_y = int(abs(sh) * h) + int(0.3 * h) + 2, int(0.3 * h) + 2
        img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        M[0, 1] += sh
        M[0, 2] -= sh * H / 2
        M = np.vstack([M, [0, 0, 1]])
        if rng.random() < 0.5:  # mild perspective (camera tilt)
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            jit = rng.uniform(-0.08, 0.08, (4, 2)) * [W * 0.25, H]
            M = cv2.getPerspectiveTransform(src, np.float32(src + jit)) @ M
        out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # trim the replicated border back to the warped content (keep a small margin)
        gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        m = np.abs(gray.astype(np.int16) - int(np.median(gray[0]))) > 12
        ys, xs = np.where(m)
        if len(xs) > 10:
            x0, x1 = max(0, xs.min() - 2), min(W, xs.max() + 3)
            y0, y1 = max(0, ys.min() - 2), min(H, ys.max() + 3)
            if x1 - x0 >= 8 and y1 - y0 >= 6:
                out = out[y0:y1, x0:x1]
        return out

    def _noise_band(self, img, bh, bw, rng):
        """Synthetic clutter: noise texture, gradient, blobs or a solid random tone."""
        shape = (bh, bw) if img.ndim == 2 else (bh, bw, img.shape[2])
        kind = rng.random()
        if kind < 0.3:
            band = rng.integers(0, 256, shape).astype(np.uint8)
            band = cv2.GaussianBlur(band, (0, 0), rng.uniform(0.5, 2.0))
        elif kind < 0.55:
            a, b = rng.integers(0, 256, 2)
            g = np.linspace(a, b, bw if rng.random() < 0.5 else bh)
            band = np.tile(g, (bh, 1)) if len(g) == bw else np.tile(g[:, None], (1, bw))
            band = np.broadcast_to(band[..., None] if img.ndim == 3 else band, shape).astype(np.uint8).copy()
        elif kind < 0.8:
            band = np.full(shape, int(rng.integers(0, 256)), np.uint8)
            for _ in range(int(rng.integers(1, 5))):
                col = int(rng.integers(0, 256)); c = col if img.ndim == 2 else (col, col, col)
                cv2.circle(band, (int(rng.integers(0, bw)), int(rng.integers(0, bh))),
                           int(rng.integers(1, max(2, min(bh, bw)))), c, -1)
        else:
            band = np.full(shape, int(rng.integers(0, 256)), np.uint8)
        if img.ndim == 3 and rng.random() < 0.7:  # a colour tint
            band = np.clip(band.astype(np.int16) + rng.integers(-60, 61, (1, 1, 3)), 0, 255).astype(np.uint8)
        return band

    _blob_renderer = None

    def ornament(self, img, rng):
        """A piece of a real COCO photo with a wiggly outline, attached BESIDE the word."""
        h, w = img.shape[:2]
        try:
            if CurveSymbolAugment._blob_renderer is None:
                from render_synth import Renderer
                CurveSymbolAugment._blob_renderer = Renderer.__new__(Renderer)
            r = CurveSymbolAugment._blob_renderer
            size = max(8, int(h * rng.uniform(0.5, 0.95)))
            side = np.empty((h, size + max(2, h // 8)) + img.shape[2:], np.uint8)
            side[:] = np.median(np.concatenate([img[:, 0], img[:, -1]]), axis=0).astype(np.uint8)
            if r._paste_blob(side, side.shape[1] // 2 + 1, h // 2, size, rng):
                return np.concatenate([side, img] if rng.random() < 0.6 else [img, side], axis=1)
        except Exception:
            pass
        size = max(8, int(h * rng.uniform(0.5, 1.0)))
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgv = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        side = np.full((h, size + 4) + img.shape[2:], bgv if img.ndim == 2 else 0, np.uint8)
        if img.ndim == 3:
            side[:] = np.median(np.concatenate([img[0], img[-1]]), axis=0).astype(np.uint8)
        col = int(rng.uniform(0, 80)) if bgv > 128 else int(rng.uniform(170, 255))
        c = col if img.ndim == 2 else tuple(int(v) for v in np.clip(np.array([col] * 3) + rng.integers(-60, 61, 3), 0, 255))
        cx, cy, r = (size + 4) // 2, h // 2, size // 2 - 1
        kind = rng.random()
        if kind < 0.3:  # flower: petals around a centre
            n = int(rng.integers(5, 9))
            for k in range(n):
                a = 2 * np.pi * k / n
                cv2.circle(side, (int(cx + 0.55 * r * np.cos(a)), int(cy + 0.55 * r * np.sin(a))), max(1, int(0.38 * r)), c, -1, cv2.LINE_AA)
            cv2.circle(side, (cx, cy), max(1, int(0.25 * r)), (255 - col) if img.ndim == 2 else tuple(255 - v for v in c), -1, cv2.LINE_AA)
        elif kind < 0.5:  # star
            pts = [(cx + (r if k % 2 == 0 else 0.45 * r) * np.cos(np.pi / 2 + k * np.pi / 5),
                    cy - (r if k % 2 == 0 else 0.45 * r) * np.sin(np.pi / 2 + k * np.pi / 5)) for k in range(10)]
            cv2.fillPoly(side, [np.array(pts, np.int32)], c, cv2.LINE_AA)
        elif kind < 0.7:  # leaf
            cv2.ellipse(side, (cx, cy), (max(1, int(0.45 * r)), r), float(rng.uniform(-40, 40)), 0, 360, c, -1, cv2.LINE_AA)
        elif kind < 0.85:  # heart
            cv2.circle(side, (cx - r // 3, cy - r // 4), max(1, r // 2), c, -1, cv2.LINE_AA)
            cv2.circle(side, (cx + r // 3, cy - r // 4), max(1, r // 2), c, -1, cv2.LINE_AA)
            cv2.fillPoly(side, [np.array([(cx - r, cy - r // 6), (cx + r, cy - r // 6), (cx, cy + r)], np.int32)], c, cv2.LINE_AA)
        else:  # a row of dots
            for k in range(3):
                cv2.circle(side, (cx, int(cy - r + k * r)), max(1, r // 4), c, -1, cv2.LINE_AA)
        return np.concatenate([side, img] if rng.random() < 0.6 else [img, side], axis=1)

    def edge_clutter(self, img, rng, ds):
        """Clutter on any of the four edges: a strip cut from another random crop (a loose crop
        or a neighbouring word) or a synthetic noisy / coloured band. Usually attached beside
        the crop; sometimes laid over the outer margin so it touches the characters."""
        h, w = img.shape[:2]
        side = int(rng.integers(4))  # 0 left, 1 right, 2 top, 3 bottom
        vert = side < 2
        size = max(2, int((w if vert else h) * rng.uniform(0.06, 0.3 if vert else 0.5)))
        bh, bw = (h, size) if vert else (size, w)
        strip = None
        if ds is not None and len(ds) >= 2 and rng.random() < 0.5:
            other = ds.raw(int(rng.integers(len(ds))))
            if other is not None and other.ndim == img.ndim and other.shape[0] >= 4 and other.shape[1] >= 8:
                oh, ow = other.shape[:2]
                if vert:
                    x0 = int(rng.integers(0, max(1, ow - max(2, ow // 4)))); part = other[:, x0:x0 + max(2, ow // 4)]
                else:
                    y0 = int(rng.integers(0, max(1, oh - max(2, oh // 3)))); part = other[y0:y0 + max(2, oh // 3)]
                strip = cv2.resize(part, (bw, bh), interpolation=cv2.INTER_AREA)
        if strip is None:
            strip = self._noise_band(img, bh, bw, rng)
        if rng.random() < 0.3:  # overlay the outer margin (may touch the characters)
            out = img.copy()
            ov = max(1, min(size, int(0.12 * (w if vert else h))))
            if rng.random() < 0.4:  # partial dark (black-ish) border along part of the edge
                dark = int(rng.integers(0, 45))
                strip = np.full_like(strip, dark)
                L = h if vert else w
                a0 = int(rng.integers(0, max(1, L // 2))); a1 = min(L, a0 + int(rng.uniform(0.2, 0.8) * L))
                if vert:
                    out[a0:a1, :ov] = strip[a0:a1, :ov] if side == 0 else out[a0:a1, :ov]
                    if side == 1: out[a0:a1, w - ov:] = strip[a0:a1, :ov]
                else:
                    if side == 2: out[:ov, a0:a1] = strip[:ov, a0:a1]
                    else: out[h - ov:, a0:a1] = strip[:ov, a0:a1]
                return out
            if side == 0: out[:, :ov] = strip[:, :ov]
            elif side == 1: out[:, w - ov:] = strip[:, :ov]
            elif side == 2: out[:ov] = strip[:ov]
            else: out[h - ov:] = strip[:ov]
            return out
        if side == 0: return np.concatenate([strip, img], axis=1)
        if side == 1: return np.concatenate([img, strip], axis=1)
        if side == 2: return np.concatenate([strip, img], axis=0)
        return np.concatenate([img, strip], axis=0)

    def lines(self, img, rng):
        """One or two long slanted lines across the full height in a text-like colour."""
        h, w = img.shape[:2]
        out = img.copy()
        gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        bg = float(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
        color = val if out.ndim == 2 else (val, val, val)
        for _ in range(int(rng.integers(1, 3))):
            x = int(rng.integers(0, w))
            dx = int(rng.uniform(-0.6, 0.6) * h)
            cv2.line(out, (x, -2), (x + dx, h + 2), color, max(1, int(rng.integers(1, max(2, h // 16)))), cv2.LINE_AA)
        return out

    def hlines(self, img, rng):
        """Horizontal rules along the top and / or bottom (underline, overline, table border);
        they may touch or cut through the characters."""
        h, w = img.shape[:2]
        out = img.copy()
        gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        bg = float(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
        color = val if out.ndim == 2 else (val, val, val)
        bands = [(0, max(1, int(0.3 * h))), (int(0.7 * h), h - 1)]
        pick = rng.random()
        for lo, hi in (bands if pick < 0.3 else [bands[0]] if pick < 0.6 else [bands[1]]):
            y = int(rng.integers(lo, max(lo + 1, hi)))
            tilt = int(rng.uniform(-0.08, 0.08) * w)
            x0 = int(rng.integers(0, max(1, w // 5))) if rng.random() < 0.4 else 0
            x1 = w - int(rng.integers(0, max(1, w // 5))) if rng.random() < 0.4 else w
            cv2.line(out, (x0, y), (x1, y + tilt), color, max(1, int(rng.integers(1, max(2, h // 14)))), cv2.LINE_AA)
        return out

    def blur(self, img, rng):
        """Strong Gaussian blur or motion blur (camera shake / defocus)."""
        if rng.random() < 0.5:
            return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 1.6))
        k = int(rng.integers(3, 8))
        kern = np.zeros((k, k), np.float32)
        ang = rng.uniform(0, np.pi)
        c = k // 2
        for t in np.linspace(-c, c, 4 * k):
            kern[min(k - 1, max(0, int(round(c + t * np.sin(ang))))), min(k - 1, max(0, int(round(c + t * np.cos(ang)))))] = 1
        kern /= kern.sum()
        return cv2.filter2D(img, -1, kern)

    def erase(self, img, rng):
        """Scrape off parts of the characters: patches / scratches in the background colour,
        stroke erosion or dilation, and speckle noise."""
        h, w = img.shape[:2]
        img = img.copy()
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        bgc = bg if img.ndim == 2 else (bg, bg, bg)
        mode = rng.random()
        if mode < 0.4:  # a few small patches
            for _ in range(int(rng.integers(1, 4))):
                pw, ph = int(rng.integers(2, max(3, w // 14))), int(rng.integers(2, max(3, h // 5)))
                x, y = int(rng.integers(0, max(1, w - pw))), int(rng.integers(0, max(1, h - ph)))
                img[y:y + ph, x:x + pw] = bgc
        elif mode < 0.7:  # scratches (thin lines)
            for _ in range(int(rng.integers(1, 4))):
                x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
                x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
                cv2.line(img, (x1, y1), (x2, y2), bgc, max(1, int(rng.integers(1, max(2, h // 18)))))
        elif mode < 0.85:  # thin or thicken the strokes
            k = np.ones((2, 2), np.uint8)
            img = cv2.erode(img, k) if (rng.random() < 0.5) == (bg > 128) else cv2.dilate(img, k)
        else:  # speckle: random pixels in the background colour
            m = rng.random((h, w)) < rng.uniform(0.01, 0.04)
            img[m] = bgc
        return img

    def arc(self, img, rng):
        """Wrap the word onto a circular arc (letters rotate along the arc): rainbow (ends
        down) or smile (ends up), with the angular span setting the curvature."""
        h, w = img.shape[:2]
        smile = rng.random() < 0.5  # ends up (smile) instead of ends down (rainbow)
        if smile:
            img = cv2.flip(img, 0)  # flipped in, flipped back out: letters stay upright
        top = getattr(self, "max_span", 150)
        # mostly moderate arcs; strong ones (up to max_span) 30% of the time, as they shrink the text
        span = np.radians(rng.uniform(25, 90) if rng.random() < 0.7 else rng.uniform(90, max(91, top)))
        R = w / span  # radius of the baseline arc
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        Ro = R + h
        out_w = int(2 * Ro * np.sin(min(span / 2, np.pi / 2))) + 4
        out_h = int(Ro - R * np.cos(span / 2)) + 4 if span < np.pi else int(Ro + R) + 4
        out_w, out_h = min(out_w, 6 * w + 8), min(out_h, 4 * h + 8)
        cx, cy = out_w / 2.0, Ro + 2.0  # centre below the canvas top; text between radii R and Ro
        X, Y = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
        dx, dy = X - cx, cy - Y
        r = np.sqrt(dx * dx + dy * dy)
        phi = np.arctan2(dx, dy)  # 0 at the top, clockwise positive
        map_x = ((phi / span) + 0.5) * w
        map_y = Ro - r
        inside = (r >= R) & (r <= Ro) & (np.abs(phi) <= span / 2)
        map_x = np.where(inside, np.clip(map_x, 0, w - 1), -10).astype(np.float32)
        map_y = np.where(inside, np.clip(map_y, 0, h - 1), -10).astype(np.float32)
        fill = bg if img.ndim == 2 else (bg, bg, bg)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        if smile:
            out = cv2.flip(out, 0)
        return out

    def partial_arc(self, img, rng):
        """Text along ONE part of a circle (e.g. a quarter): it starts near the top of the circle
        and runs down one side, so it goes from top-left to bottom-right (or the mirror image,
        or the same below the centre)."""
        below = rng.random() < 0.35  # text on the lower part of the circle (letters stay upright)
        if below:
            img = cv2.flip(img, 0)
        h, w = img.shape[:2]
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        span = np.radians(rng.uniform(45, 110))
        R = w / span
        Ro = R + h
        if rng.random() < 0.6:  # falling: starts at the top of the circle, runs down to the right
            a0 = np.radians(rng.uniform(-15, 10))
        else:  # rising: starts on the left side, runs up to the top
            a0 = -span + np.radians(rng.uniform(-10, 15))
        a1 = a0 + span
        # bounding box of the band in circle coordinates (centre at 0, 0; y up)
        ang = np.linspace(a0, a1, 64)
        xs = np.concatenate([R * np.sin(ang), Ro * np.sin(ang)])
        ys = np.concatenate([R * np.cos(ang), Ro * np.cos(ang)])
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        out_w, out_h = int(x_max - x_min) + 4, int(y_max - y_min) + 4
        if out_w > 8 * w + 8 or out_h > 8 * h + 8 or out_w < 4 or out_h < 4:
            return img
        X, Y = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
        cx_, cy_ = X + x_min - 2, (y_max + 2) - Y  # pixel -> circle coordinates (y up)
        r = np.sqrt(cx_ * cx_ + cy_ * cy_)
        phi = np.arctan2(cx_, cy_)  # clockwise from the top
        map_x = (phi - a0) / span * w
        map_y = Ro - r
        inside = (r >= R) & (r <= Ro) & (phi >= a0) & (phi <= a1)
        map_x = np.where(inside, np.clip(map_x, 0, w - 1), -10).astype(np.float32)
        map_y = np.where(inside, np.clip(map_y, 0, h - 1), -10).astype(np.float32)
        fill = bg if img.ndim == 2 else (bg, bg, bg)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        return cv2.flip(out, 0) if below else out

    def bend(self, img, rng):
        if rng.random() < getattr(self, "partial_share", 0.25):
            return self.partial_arc(img, rng)
        if rng.random() < getattr(self, "arc_share", 0.6):  # true arc vs. parabola / wave
            return self.arc(img, rng)
        h, w = img.shape[:2]
        amp = rng.uniform(0.12, self.max_bend) * h * rng.choice([-1, 1])
        pad = int(abs(amp)) + 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_REPLICATE)
        H = img.shape[0]
        xs = np.arange(w, dtype=np.float32)
        t = (xs - w / 2) / (w / 2)  # -1 .. 1
        if rng.random() < 0.75:
            shift = amp * (t * t - 0.5)  # arc: ends up / middle down (or the reverse)
        else:
            shift = amp * 0.7 * np.sin(np.pi * t * rng.uniform(0.6, 1.2) + rng.uniform(0, 6.28))
        map_x = np.tile(xs, (H, 1))
        map_y = (np.arange(H, dtype=np.float32)[:, None] + shift[None, :]).astype(np.float32)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    def symbols(self, img, rng):
        h, w = img.shape[:2]
        if h < 8 or w < 8:
            return img
        img = img.copy()
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n = int(rng.integers(1, 4))
        for _ in range(n):
            if rng.random() < 0.3:  # dot-like marks of varied size / thickness (not an "o")
                self._dot(img, gray, rng)
                continue
            if rng.random() < 0.25:  # a box around the whole word or around a few characters
                self._box(img, gray, rng)
                continue
            if rng.random() < 0.3:  # drawn exclamation mark, varied proportions
                self._exclaim(img, gray, rng)
                continue
            sym = self.SYMS[int(rng.integers(len(self.SYMS)))]
            font = self.FONTS[int(rng.integers(len(self.FONTS)))] | (cv2.FONT_ITALIC if rng.random() < 0.3 else 0)
            scale = rng.uniform(0.35, 0.8) * h / 22.0  # Hershey font height ~22 px at scale 1
            thick = max(1, int(round(scale * rng.uniform(1.0, 2.5))))
            (tw, th), _ = cv2.getTextSize(sym, font, scale, thick)
            th, tw = min(th, h - 1), min(tw, w - 1)  # tiny crops: keep the ranges valid
            where = rng.random()
            if where < 0.2:  # start
                x = int(rng.integers(0, max(1, min(w // 6, w - tw))))
                y = int(rng.integers(th, h))
            elif where < 0.4:  # end
                x = int(rng.integers(max(0, w - tw - w // 6), max(1, w - tw)))
                y = int(rng.integers(th, h))
            elif where < 0.88:  # between characters, anywhere from the middle to the baseline
                x = int(rng.integers(0, max(1, w - tw)))
                y = int(rng.integers(int(0.4 * h), h))
            else:  # above the text
                x = int(rng.integers(0, max(1, w - tw)))
                y = int(rng.integers(th, max(th + 1, int(0.4 * h))))
            x0, y0 = max(0, x), max(0, y - th)
            local = gray[y0:min(h, y0 + th + 2), x0:min(w, x0 + tw + 2)]
            bg = float(local.mean()) if local.size else 128.0
            # contrast against the local background (like real text / symbols)
            val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
            color = val if img.ndim == 2 else (val, val, val)
            cv2.putText(img, sym, (x, y), font, scale, color, thick, cv2.LINE_AA)
        return img

    def _box(self, img, gray, rng):
        h, w = img.shape[:2]
        bg = float(gray.mean())
        val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
        color = val if img.ndim == 2 else (val, val, val)
        t = max(1, int(rng.integers(1, max(2, h // 10))))
        if rng.random() < 0.5:  # whole word, inset by a small margin (may touch the characters)
            mx, my = int(rng.uniform(0, 0.08) * w), int(rng.uniform(0, 0.15) * h)
            x0, y0, x1, y1 = mx, my, w - 1 - mx, h - 1 - my
        else:  # a span of a few characters
            x0 = int(rng.integers(0, max(1, int(0.7 * w)))); x1 = min(w - 1, x0 + int(rng.uniform(0.15, 0.5) * w))
            y0 = int(rng.uniform(0, 0.2) * h); y1 = h - 1 - int(rng.uniform(0, 0.2) * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, t)

    FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
             cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

    def _exclaim(self, img, gray, rng):
        """An exclamation mark drawn as a bar + dot with varied proportions / thickness, mostly
        right after the word (where real "!" marks are), sometimes at the start."""
        h, w = img.shape[:2]
        bar_h = int(rng.uniform(0.45, 0.95) * h)
        t = max(1, int(rng.uniform(0.04, 0.16) * h))
        gap = max(1, int(rng.uniform(0.05, 0.15) * h))
        if rng.random() < 0.8:  # after the word
            lo = max(0, min(w - t - 1, w - int(0.18 * w)))
            x = int(rng.integers(lo, max(lo + 1, w - t)))
        else:
            x = int(rng.integers(0, max(1, int(0.12 * w))))
        y_top = int(rng.uniform(0, 0.25) * h)
        local = gray[:, max(0, x - 2 * t):x + 2 * t + 1]
        bg = float(local.mean()) if local.size else 128.0
        val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
        color = val if img.ndim == 2 else (val, val, val)
        y_bar_end = min(h - 2 * t - gap - 1, y_top + bar_h)
        if rng.random() < 0.5:  # tapered bar
            pts = np.array([[x, y_top], [x + t, y_top], [x + max(1, t // 2) + max(0, t // 4), y_bar_end], [x + max(0, t // 4), y_bar_end]], np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y_top), (x + t, y_bar_end), color, -1)
        r = max(1, int(t * rng.uniform(0.5, 0.9)))
        cv2.circle(img, (x + t // 2, min(h - 1, y_bar_end + gap + r)), r, color, -1, cv2.LINE_AA)

    def _dot(self, img, gray, rng):
        h, w = img.shape[:2]
        r = max(1, int(rng.uniform(1, max(1.5, h / 5))))
        zone = rng.random()
        y = int(rng.integers(int(0.7 * h), h)) if zone < 0.55 else int(rng.integers(0, h)) if zone < 0.8 else int(rng.integers(0, max(1, int(0.35 * h))))
        x = int(rng.integers(0, w))
        local = gray[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        bg = float(local.mean()) if local.size else 128.0
        val = int(rng.uniform(0, 70)) if bg > 128 else int(rng.uniform(185, 255))
        color = val if img.ndim == 2 else (val, val, val)
        kind = rng.random()
        if kind < 0.5:
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
        elif kind < 0.7:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)
        elif kind < 0.85:  # small ring
            cv2.circle(img, (x, y), max(2, r), color, max(1, r // 2), cv2.LINE_AA)
        else:  # double dot (colon-like or ellipsis start)
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
            cv2.circle(img, (min(w - 1, x + 3 * r + 1), y), r, color, -1, cv2.LINE_AA)

    MAX_OPS = 2  # at most this many heavy effects per image, so crops stay readable
    MIN_TEXT_PX = 20  # the word's height (the crop height) after the fit to 48 x 192

    def __call__(self, img, i, ds=None):
        """Apply the effects; if the result would shrink the word below a readable size once it
        is fitted to the model input (48 x 192), redo it without the size-changing effects."""
        h0 = img.shape[0]
        out = self._apply(img, i, ds)
        H, W = out.shape[:2]
        if h0 * min(48.0 / max(1, H), 192.0 / max(1, W)) >= self.MIN_TEXT_PX:
            return out
        return self._apply(img, i, ds, size_safe=True)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

    def _apply(self, img, i, ds=None, size_safe=False):
        rng = np.random.default_rng()
        ops = []
        curved_text = bool(getattr(ds, "_no_bend", False))
        geom = rng.random() < self.p_geom  # geometry (slant / rotation, bend) is never capped
        harmless = []  # ops allowed on geometrically transformed (curved / rotated) words
        if rng.random() < self.p_edge and not size_safe:
            def _edge(im):
                if rng.random() < 0.3:  # a decoration beside the word instead of clutter
                    return self.ornament(im, rng)
                im = self.edge_clutter(im, rng, ds)
                return self.edge_clutter(im, rng, ds) if rng.random() < 0.4 else im  # a second side
            ops.append(_edge); harmless.append(_edge)
        if rng.random() < self.p_lines:
            ops.append(lambda im: self.lines(im, rng))
        if rng.random() < self.p_lines:
            ops.append(lambda im: self.hlines(im, rng))
        if rng.random() < self.p_symbols:
            ops.append(lambda im: self.symbols(im, rng)); harmless.append(ops[-1])
        bend = (not curved_text) and rng.random() < self.p_bend
        if size_safe:
            geom = bend = False
        if bend or curved_text:  # keep curved words readable: no extra rotation, <= 1 more effect
            geom = False
        if rng.random() < self.p_erase:
            ops.append(lambda im: self.erase(im, rng))
        if rng.random() < self.p_tone:
            ops.append(lambda im: self.tone(im, rng)); harmless.append(ops[-1])
        if rng.random() < self.p_blur:
            ops.append(lambda im: self.blur(im, rng))
        max_ops = getattr(self, "max_ops", self.MAX_OPS)
        if bend or curved_text or geom:  # keep geometric words clean: no blur / scrape / lines
            ops = [op for op in ops if any(op is h for h in harmless)]
            max_ops = min(max_ops, 1)
        if len(ops) > max_ops:  # keep a random subset, in the original order
            keep = set(rng.choice(len(ops), max_ops, replace=False).tolist())
            ops = [op for j, op in enumerate(ops) if j in keep]
        if geom:
            img = self.geom(img, rng)
        for op in ops:
            img = op(img)
        if bend:  # last, so symbols / lines / boxes are bent together with the word
            img = self.bend(img, rng)
        return img


class ExploreAugment:
    """Augmentation of exploreSTR-wide (data.py): mild affine with shear, blur, low-resolution
    simulation, polarity, contrast/brightness, noise. p_scale multiplies every probability
    (e.g. 0.5 for rendered words, which are already degraded)."""

    def __init__(self, p_scale=1.0):
        self.p_scale = p_scale

    def __call__(self, img, i):
        rng = np.random.default_rng()
        h, w = img.shape[:2]
        ps = self.p_scale
        if rng.random() < 0.7 * ps:  # mild affine
            a, sh, sc = rng.uniform(-4, 4), rng.uniform(-0.25, 0.25), rng.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, sc)
            M[0, 1] += sh
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.3 * ps:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.3 * ps:  # downsample-upsample (low-resolution real crops)
            f = rng.uniform(0.3, 0.7)
            img = cv2.resize(cv2.resize(img, (max(4, int(w * f)), max(4, int(h * f)))), (w, h))
        if rng.random() < 0.15 * ps:
            img = 255 - img
        if rng.random() < 0.5 * ps:
            img = np.clip(img.astype(np.float32) * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40),
                          0, 255).astype(np.uint8)
        if rng.random() < 0.2 * ps:
            img = np.clip(img + rng.normal(0, rng.uniform(2, 12), img.shape), 0, 255).astype(np.uint8)
        return img


class Augment:
    """CRNN-style augmentation for grayscale word crops (training only)."""

    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, img, i):
        out = self._augment(img)
        if out is None or out.ndim != 2 or out.shape[0] < 4 or out.shape[1] < 4:
            return img  # tiny crops: keep the original
        return out

    def _augment(self, img):
        rng = np.random.default_rng()
        if rng.random() > self.p:
            return img
        h, w = img.shape[:2]
        # crop jitter: move each border in or out by up to 8% of the height
        d = max(1, int(0.08 * h))
        top, bot, left, right = (int(v) for v in rng.integers(-d, d + 1, 4))
        img = cv2.copyMakeBorder(img, max(0, -top), max(0, -bot), max(0, -left), max(0, -right),
                                 cv2.BORDER_REPLICATE)
        H, W = img.shape[:2]
        img = img[max(0, top):H - max(0, bot), max(0, left):W - max(0, right)]
        h, w = img.shape[:2]
        if h < 4 or w < 4:
            return None
        # small rotation + perspective
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jit = rng.uniform(-0.06, 0.06, (4, 2)) * [w * 0.3, h]
        M = cv2.getPerspectiveTransform(src, np.float32(src + jit))
        ang = rng.uniform(-4, 4)
        R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), [0, 0, 1]])
        img = cv2.warpPerspective(img, R @ M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
        img = img.astype(np.float32)
        if rng.random() < 0.5:  # contrast / brightness
            img = (img - img.mean()) * rng.uniform(0.5, 1.5) + img.mean() + rng.uniform(-30, 30)
        if rng.random() < 0.3:
            img = img + rng.normal(0, rng.uniform(2, 12), img.shape)
        if rng.random() < 0.2:  # text polarity
            img = 255 - img
        return np.clip(img, 0, 255).astype(np.uint8)


def load_alphabet(spec):
    """--alphabet value: the characters themselves, or a text file holding them
    (all lines joined, newlines removed)."""
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            spec = "".join(line.rstrip("\n") for line in f)
    chars = "".join(dict.fromkeys(spec))  # drop repeats, keep order
    if not chars:
        raise ValueError("empty alphabet")
    return chars


def _clean(label, alphabet, max_len):
    if alphabet is None:  # evaluation: keep every sample unchanged
        return label.strip()
    word = label.strip()
    if not any(ch != ch.lower() for ch in alphabet):
        word = word.lower()
    if not word or len(word) > max_len or any(ch not in alphabet for ch in word):
        return None
    return word


_LMDB_ENVS = {}


def _open_lmdb(path):
    """One read-only environment per path and process (LMDB refuses a second open)."""
    import lmdb

    key = (os.getpid(), os.path.realpath(path))
    if key not in _LMDB_ENVS:
        _LMDB_ENVS[key] = lmdb.open(path, readonly=True, lock=False,
                                    readahead=False, meminit=False)
    return _LMDB_ENVS[key]


class LmdbDataset(Dataset):
    """Standard text-recognition LMDB; the environment is opened lazily per worker."""

    def __init__(self, path, img_size=(32, 96), alphabet=None, max_len=25, keys=None,
                 labels=None):
        import lmdb

        self.path, self.img_size, self.env = path, img_size, None
        self.alphabet, self.max_len = alphabet, max_len
        self.labels = labels or {}  # key -> label override (e.g. symbols stripped)
        if keys is not None:  # precomputed by tools/prep_textnet_data.py
            self.index = list(keys)
            return
        env = _open_lmdb(path)
        with env.begin() as txn:
            n = int(txn.get(b"num-samples"))
            self.index = []
            for i in range(1, n + 1):
                lab = txn.get(f"label-{i:09d}".encode())
                if lab is not None and (alphabet is None or
                                        _clean(lab.decode("utf-8", "ignore"), alphabet, max_len)):
                    self.index.append(i)

    def __len__(self):
        return len(self.index)

    def raw(self, i):
        """Decoded crop without any transform (used by the edge-clutter augmentation)."""
        if self.env is None:
            self.env = _open_lmdb(self.path)
        with self.env.begin() as txn:
            buf = txn.get(f"image-{self.index[i]:09d}".encode())
        flag = cv2.IMREAD_COLOR if getattr(self, "color", False) else cv2.IMREAD_GRAYSCALE
        return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)

    def __getitem__(self, i):
        import lmdb

        if self.env is None:
            self.env = _open_lmdb(self.path)
        k = self.index[i]
        with self.env.begin() as txn:
            buf = txn.get(f"image-{k:09d}".encode())
            lab = self.labels.get(k) if self.labels else None
            if lab is None:
                lab = txn.get(f"label-{k:09d}".encode()).decode("utf-8", "ignore")
        flag = cv2.IMREAD_COLOR if getattr(self, "color", False) else cv2.IMREAD_GRAYSCALE
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), flag)
        return prepare(self, img, i), _clean(lab, self.alphabet, self.max_len)


class RenderDataset(Dataset):
    """Words rendered on the fly (tools/render_synth.py); every access is a new random image.
    The label is the alnum core of what is drawn (symbols are drawn but not labelled)."""

    def __init__(self, n, img_size=(32, 96), alphabet=None, max_len=25):
        self.n, self.img_size, self.alphabet, self.max_len = n, img_size, alphabet, max_len
        self._r = None

    def _renderer(self):
        if self._r is None:
            from render_synth import Renderer, load_lexicon

            self._r = Renderer(load_lexicon(), color=True)
            self._r.p_arc_text = getattr(self, "p_arc_text", 0.6)  # letters drawn along an arc
            self._r.p_segment = getattr(self, "p_segment", 0.1)  # 7 / 14-segment calculator / LED displays
            self._r.p_plaque = float(os.environ.get("RENDER_PLAQUE", "0"))  # house-name plaque rim style (OFF: user 2026-10-11 09:35)
            self._r.p_logo_arc = getattr(self, "p_logo_arc", 0.25)  # round-sign style (big light letters on dark, decoration inside the curve)
        return self._r

    def __len__(self):
        return self.n

    def raw(self, i):
        img, _ = self._renderer().render()
        return img if getattr(self, "color", False) else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __getitem__(self, i):
        r = self._renderer()
        img, lab = r.render()
        self._no_bend = bool(getattr(r, "last_arc", False))  # already curved: no image warp
        if not getattr(self, "color", False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = _clean(lab, self.alphabet, self.max_len) if self.alphabet else lab
        if not lab:  # outside the alphabet (rare): draw another
            return self[i]
        return prepare(self, img, i), lab


def build_dataset(root, src, alphabet, max_len=25, img_size=(32, 96)):
    """Label file, LMDB folder, or folder of LMDB folders -> Dataset."""
    from torch.utils.data import ConcatDataset

    kw = dict(img_size=img_size, alphabet=alphabet, max_len=max_len)
    path = src if os.path.isabs(src) else os.path.join(root, src)
    if path.endswith(".keys.json"):  # key lists from tools/prep_textnet_data.py
        with open(path) as f:
            spec = json.load(f)
        if spec.get("render"):  # on-the-fly synthetic renderer
            return RenderDataset(int(spec.get("n", 100000)), **kw)
        over = spec.get("labels", {})
        return ConcatDataset([LmdbDataset(p, keys=k, labels={int(a): b for a, b in over.get(p, {}).items()}, **kw)
                              for p, k in spec["sources"].items()])
    if os.path.isfile(os.path.join(path, "data.mdb")):
        return LmdbDataset(path, **kw)
    if os.path.isdir(path):
        subs = sorted(os.path.join(path, d) for d in os.listdir(path)
                      if os.path.isfile(os.path.join(path, d, "data.mdb")))
        return ConcatDataset([LmdbDataset(p, **kw) for p in subs])
    return TextLineDataset(root, src, **kw)


class TextLineDataset(Dataset):
    def __init__(self, root, label_file, img_size=(32, 96), alphabet=None, max_len=25):
        self.root, self.img_size = root, img_size
        self.samples = []
        with open(os.path.join(root, label_file)) as f:
            for line in f:
                parts = line.rstrip("\n").split(maxsplit=1)
                if len(parts) != 2:
                    continue
                word = parts[1] if alphabet is None else _clean(parts[1], alphabet, max_len)
                if word:
                    self.samples.append((parts[0], word))

    def __len__(self):
        return len(self.samples)

    def raw(self, i):
        flag = cv2.IMREAD_COLOR if getattr(self, "color", False) else cv2.IMREAD_GRAYSCALE
        return cv2.imread(os.path.join(self.root, self.samples[i][0]), flag)

    def __getitem__(self, i):
        path, text = self.samples[i]
        flag = cv2.IMREAD_COLOR if getattr(self, "color", False) else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(os.path.join(self.root, path), flag)
        return prepare(self, img, i), text


def save_atomic(obj, path):
    """torch.save via a temp file, so a crash never leaves a truncated checkpoint."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def collate(batch):
    imgs, texts = zip(*batch)
    return torch.stack(imgs), list(texts)


@torch.no_grad()
def evaluate(model, loader, device, beam_size=1):
    model.eval()
    correct = total = 0
    for imgs, texts in loader:
        preds = model.recognize(imgs.to(device), beam_size=beam_size)
        for p, t in zip(preds, texts):
            gt = "".join(ch for ch in model.normalize(t) if ch in model.alphabet)
            correct += int(p == gt)
            total += 1
    model.train()
    if getattr(model, "_frozen_bb", False):
        model.backbone.backbone.eval()
    return correct / max(total, 1)


OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
}

# Learning-rate schedules, as a factor of --lr per iteration (after --warmup):
#   none       constant (no warm-up)
#   cosine     cosine decay to --min-lr-ratio
#   linear     linear decay to --min-lr-ratio
#   poly       (1 - t)^--power decay to --min-lr-ratio
#   step       x --gamma every --step-size iterations
#   multistep  x --gamma at each of --milestones
#   exp        smooth exponential decay, x --gamma per --step-size iterations
#   plateau    constant, x --gamma after --patience evaluations without a new best
SCHEDULERS = ["none", "cosine", "linear", "poly", "step", "multistep", "exp", "plateau"]


def build_optimizer(params, args):
    name = args.optimizer
    kw = {"lr": args.lr, "weight_decay": args.weight_decay}
    if name == "sgd":
        m = 0.9 if args.momentum is None else args.momentum
        kw.update(momentum=m, nesterov=args.nesterov and m > 0)
    elif name == "rmsprop":
        kw.update(momentum=args.momentum or 0.0)
    return OPTIMIZERS[name](params, **kw)


def lr_factor(step, total, args):
    """Multiplier of --lr at iteration `step` (warm-up included)."""
    import bisect
    import math

    s = args.sched
    if s == "none":
        return 1.0
    if step < args.warmup:
        return (step + 1) / args.warmup
    k = step - args.warmup
    t = min(1.0, k / max(1, total - args.warmup))
    lo = args.min_lr_ratio
    if s == "cosine":
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * t))
    if s == "linear":
        return lo + (1 - lo) * (1 - t)
    if s == "poly":
        return lo + (1 - lo) * (1 - t) ** args.power
    if s == "step":
        return args.gamma ** (k // args.step_size)
    if s == "multistep":
        return args.gamma ** bisect.bisect_right(sorted(args.milestones), step)
    if s == "exp":
        return args.gamma ** (k / args.step_size)
    return 1.0  # plateau: constant here, decayed on evaluation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s", choices=VARIANTS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--train", nargs="*", default=[], help="training sources (concatenated)")
    ap.add_argument("--train-weights", type=float, nargs="*", default=None,
                    help="share of each --train source in every batch (sampled with "
                         "replacement); default: proportional to source size")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze the CSPDarknet backbone (weights and BN statistics); train the rest")
    ap.add_argument("--init-skip", default=None,
                    help="regex of tensor names NOT to load from --init (e.g. 'qge\\.alpha$' "
                         "to keep a fixed QGE alpha)")
    ap.add_argument("--init", default=None,
                    help="weights to start from; tensors whose name or shape does not "
                         "match the model are skipped (e.g. after changing --img-size)")
    ap.add_argument("--val", nargs="*", default=[], help="dev files (model selection)")
    ap.add_argument("--test", nargs="*", default=[], help="final evaluation files")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--iters", type=int, default=None, help="train by iterations instead of epochs")
    ap.add_argument("--eval-interval", type=int, default=2000, help="iterations (with --iters)")
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--img-size", type=int, nargs=2, default=[32, 96], metavar=("H", "W"),
                    help="input height and width (multiples of 16)")
    ap.add_argument("--alphabet", default="0123456789abcdefghijklmnopqrstuvwxyz",
                    help="character set: the characters themselves or a text file holding "
                         "them; labels are case-sensitive if it contains upper-case letters")
    ap.add_argument("--amp", action="store_true", help="mixed precision (fp16) training")
    ap.add_argument("--optimizer", default="rmsprop", choices=sorted(OPTIMIZERS),
                    help="adadelta usually needs --lr 1.0")
    ap.add_argument("--momentum", type=float, default=None,
                    help="momentum for sgd (default 0.9) and rmsprop (default 0)")
    ap.add_argument("--nesterov", action="store_true", help="Nesterov momentum (sgd)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--sched", default="none", choices=SCHEDULERS,
                    help="learning-rate schedule over iterations; see SCHEDULERS in this file")
    ap.add_argument("--warmup", type=int, default=1000,
                    help="linear warm-up iterations (every --sched except none)")
    ap.add_argument("--min-lr-ratio", type=float, default=0.01,
                    help="final lr / --lr for cosine, poly and linear")
    ap.add_argument("--step-size", type=int, default=10000,
                    help="iterations between decays (step, exp)")
    ap.add_argument("--milestones", type=int, nargs="+", default=[20000, 30000],
                    help="iterations at which the lr is multiplied by --gamma (multistep)")
    ap.add_argument("--gamma", type=float, default=0.1,
                    help="decay factor (step, multistep, exp, plateau)")
    ap.add_argument("--power", type=float, default=0.9, help="exponent (poly)")
    ap.add_argument("--patience", type=int, default=3,
                    help="evaluations without improvement before decaying (plateau)")
    ap.add_argument("--augment", action="store_true", help="training augmentation (train only)")
    ap.add_argument("--aug-type", default="crnn", choices=["crnn", "explore"],
                    help="crnn: built-in CRNN-style; explore: exploreSTR-wide augmentation")
    ap.add_argument("--in-chans", type=int, default=1, choices=[1, 3],
                    help="1: grayscale input (default); 3: colour (BGR) input")
    ap.add_argument("--aug-bend", type=float, default=0.0,
                    help="training: probability of bending the text along an arc / wave")
    ap.add_argument("--aug-symbols", type=float, default=0.0,
                    help="training: probability of drawing punctuation / symbols into the crop")
    ap.add_argument("--aug-blur", type=float, default=0.0,
                    help="training: probability of strong Gaussian / motion blur")
    ap.add_argument("--aug-erase", type=float, default=0.0,
                    help="training: probability of scraping off parts of the characters")
    ap.add_argument("--aug-edge", type=float, default=0.0,
                    help="training: probability of pasting a strip of another crop on an edge")
    ap.add_argument("--aug-lines", type=float, default=0.0,
                    help="training: probability of long slanted lines across the crop")
    ap.add_argument("--aug-geom", type=float, default=0.0,
                    help="training: probability of strong slant (shear), rotation and tilt")
    ap.add_argument("--aug-tone", type=float, default=0.0,
                    help="training: probability of per-character brightness / colour variation")
    ap.add_argument("--no-keep-ratio", dest="keep_ratio", action="store_false",
                    help="plain resize (stretch) to --img-size; overrides an earlier --keep-ratio")
    ap.add_argument("--keep-ratio", action="store_true",
                    help="aspect-preserving resize, left-aligned and padded (train and eval) "
                         "instead of stretching to --img-size")
    ap.add_argument("--resume", default=None, help="last_state.pth: model + optimizer + step")
    ap.add_argument("--start-step", type=int, default=0,
                    help="with --ckpt (weights only): continue the schedule from this step")
    ap.add_argument("--stop-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="stop as soon as the dev accuracy (in %%) lands in [LO, HI]; that "
                         "checkpoint is saved as target.pth")
    ap.add_argument("--keep-from", type=float, default=None,
                    help="save every evaluated checkpoint with dev accuracy (%%) >= this as "
                         "ckpt_iter<N>_<acc>.pth (default: LO - 1 with --stop-window)")
    ap.add_argument("--fine-eval-from", type=float, default=None,
                    help="once dev accuracy (%%) reaches this, evaluate every "
                         "--fine-eval-interval iterations (default: LO - 1.5 with --stop-window)")
    ap.add_argument("--fine-eval-interval", type=int, default=250)
    ap.add_argument("--save-interval", type=int, default=500,
                    help="iterations between resume-state saves (0: only at evaluations)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="stop training after this many hours (then final evaluation)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-quality", action="store_true", help="ablation: no QGE quality branch")
    ap.add_argument("--no-afw", action="store_true", help="ablation: no AFW")
    ap.add_argument("--qge-type", default="global", choices=["global", "local"],
                    help="QGE quality branch: global average pooling (one value per channel) "
                         "or local k x k average (one value per channel and position)")
    ap.add_argument("--qge-local-kernel", type=int, default=7,
                    help="window size for --qge-type local (odd)")
    ap.add_argument("--qge-no-spatial", action="store_true",
                    help="ablation: QGE without the spatial attention map (quality gate only)")
    ap.add_argument("--qge-normal-gate", action="store_true",
                    help="use the gate w instead of the inverted gate 1 - w")
    ap.add_argument("--qge-alpha", default="free", choices=["free", "bounded", "fixed"],
                    help="QGE blend factor: learnable unbounded, learnable in (0, 1), or fixed")
    ap.add_argument("--qge-alpha-init", type=float, default=0.5, help="initial (or fixed) alpha")
    ap.add_argument("--qge-reduction", type=int, default=8, help="QGE FC hidden = max(C / r, 8)")
    ap.add_argument("--qge-spatial-kernel", type=int, default=7, help="QGE spatial kernel (odd)")
    ap.add_argument("--qge-gate", default="sigmoid", choices=["sigmoid", "free"],
                    help="QGE gate: sigmoid (in (0, 1), output scaled within [1 - alpha, 1]) or "
                         "free (no sigmoid: raw scores, scale per channel/position unconstrained)")
    ap.add_argument("--bottleneck-dilation", type=int, default=2,
                    help="dilation of the bottleneck's depthwise path")
    ap.add_argument("--act", default="silu", choices=["silu", "relu", "lrelu"],
                    help="activation in the backbone and neck")
    ap.add_argument("--ema", action="store_true",
                    help="keep an exponential moving average of the weights for evaluation")
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="max gradient norm (0 disables clipping)")
    ap.add_argument("--beam-size", type=int, default=1,
                    help="CTC decoding beam width (1 = greedy)")
    ap.add_argument("--gce", action="store_true",
                    help="Global Context Encoder: Transformer layers on the stride-16 "
                         "(coarsest) backbone map before the neck")
    ap.add_argument("--gce-layers", type=int, default=1)
    ap.add_argument("--gce-heads", type=int, default=8)
    ap.add_argument("--gce-ff-ratio", type=float, default=2.0)
    ap.add_argument("--afw-type", default="mlp", choices=["mlp", "deep", "attn"],
                    help="AFW weight predictor: plain MLP, deep (wider projection + residual "
                         "LayerNorm MLP blocks; e.g. --afw-layers 6 --afw-hidden 1024 to test "
                         "a much bigger AFW) or self-attention over the levels")
    ap.add_argument("--afw-layers", type=int, default=3,
                    help="AFW MLP layers, residual blocks with --afw-type deep, or "
                         "self-attention layers with --afw-type attn")
    ap.add_argument("--afw-hidden", type=int, default=None,
                    help="AFW MLP hidden units (default: sum of level channels / 4); width with "
                         "--afw-type deep (default: sum of level channels); feed-forward size "
                         "with --afw-type attn (default: 2 x --afw-attn-dim)")
    ap.add_argument("--afw-weight", default="softmax", choices=["softmax", "sigmoid", "free"],
                    help="how AFW scores become level weights: softmax (sum to 1, levels share a "
                         "fixed total), sigmoid (independent gate in (0, 1) per level), or free "
                         "(each level scaled by 1 + score: no normalization, floor or cap)")
    ap.add_argument("--afw-min-weight", type=float, default=0.0,
                    help="floor on each AFW level weight: w = m + (1 - n*m) * softmax "
                         "(0 = original AFW; e.g. 0.2 keeps every level at 0.2-0.6)")
    ap.add_argument("--afw-base", type=float, nargs="+", default=None,
                    help="fixed base weight per level (sum 1), e.g. 0.571 0.286 0.143; AFW then "
                         "only decides --afw-share of the total")
    ap.add_argument("--afw-share", type=float, default=0.1,
                    help="with --afw-base: w = (1 - share) * base + share * softmax")
    ap.add_argument("--afw-per-column", action="store_true",
                    help="AFW level weights per column (width position) instead of per image")
    ap.add_argument("--afw-norm-input", action="store_true",
                    help="parameter-free LayerNorm of each level's pooled vector before AFW")
    ap.add_argument("--level-dropout", type=float, default=0.0,
                    help="training only, --level-fusion channel / fpn / add / asff / sk: fraction of images with one "
                         "random level zeroed")
    ap.add_argument("--afw-attn-dim", type=int, default=128, help="AFW token size (attn)")
    ap.add_argument("--afw-heads", type=int, default=4, help="AFW attention heads (attn)")
    # sequence model (defaults: 2 x BLSTM(256) + CTC)
    ap.add_argument("--encoder", default="blstm", choices=["blstm", "transformer", "none"],
                    help="sequence encoder after the backbone")
    ap.add_argument("--enc-layers", type=int, default=2,
                    help="number of BLSTM layers or Transformer encoder layers")
    ap.add_argument("--hidden", type=int, default=256, help="BLSTM hidden units per direction")
    ap.add_argument("--embed-dim", type=int, default=64,
                    help="size of the projected backbone features (Transformer encoder model size)")
    ap.add_argument("--decoder", default="ctc", choices=["ctc", "attn"],
                    help="ctc, or attn = Transformer attention decoder (cross-entropy)")
    ap.add_argument("--dec-layers", type=int, default=2, help="attention decoder layers")
    ap.add_argument("--dec-dim", type=int, default=256, help="attention decoder model size")
    ap.add_argument("--num-heads", type=int, default=4, help="attention heads (encoder and decoder)")
    ap.add_argument("--ff-dim", type=int, default=1024, help="Transformer feed-forward size")
    ap.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    ap.add_argument("--level-fusion", default="time",
                    choices=["time", "channel", "fpn", "add", "asff", "sk"],
                    help="time: levels concatenated along time (original); channel: strides 8 "
                         "and 16 upsampled to the stride-4 width and concatenated per position; "
                         "fpn: levels on the stride-4 grid, concat + 1x1 conv, residual; "
                         "add: sum of the levels on the stride-4 grid + 3x3 conv; "
                         "asff: per-position softmax weights over the levels (1x1 conv), "
                         "weighted sum + 3x3 conv; sk: per-channel softmax weights over the "
                         "levels (GAP + FC), weighted sum + 3x3 conv")
    ap.add_argument("--height-pool", default="none", choices=["none", "avg", "max", "conv"],
                    help="height -> 1 per column: flatten C x H (none), pool over height (avg / "
                         "max) or CRNN-style strided 3x1 convs (conv, grid fusions only)")
    ap.add_argument("--fpn-upsample", default="bilinear", choices=["bilinear", "shuffle", "deconv"],
                    help="fpn / add / asff / sk fusion: how strides 8 / 16 reach the stride-4 grid")
    ap.add_argument("--rnn-dropout", type=float, default=0.0, help="dropout between BLSTM layers")
    ap.add_argument("--metrics-out", default=None, help="json with final test accuracies")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth, width = VARIANTS[args.variant]
    model_cfg = dict(
        depth=depth, width=width, img_size=tuple(args.img_size), in_chans=args.in_chans,
        alphabet=load_alphabet(args.alphabet),
        use_quality=not args.no_quality, use_afw=not args.no_afw, act=args.act,
        qge_cfg=dict(qge_type=args.qge_type, local_kernel=args.qge_local_kernel,
                     invert=not args.qge_normal_gate, alpha_mode=args.qge_alpha,
                     alpha_init=args.qge_alpha_init, reduction=args.qge_reduction,
                     spatial_kernel=args.qge_spatial_kernel, dilation=args.bottleneck_dilation,
                     gate_mode=args.qge_gate,
                     **({"use_spatial": False} if args.qge_no_spatial else {})),
        gce_cfg=dict(num_layers=args.gce_layers, num_heads=args.gce_heads,
                     ff_ratio=args.gce_ff_ratio) if args.gce else None,
        afw_cfg=dict(afw_type=args.afw_type, num_layers=args.afw_layers, hidden=args.afw_hidden,
                     attn_dim=args.afw_attn_dim, num_heads=args.afw_heads,
                     min_weight=args.afw_min_weight, norm_input=args.afw_norm_input,
                     base=args.afw_base, share=args.afw_share, per_column=args.afw_per_column,
                     weight_mode=args.afw_weight),
        encoder=args.encoder, enc_layers=args.enc_layers, hidden=args.hidden,
        embed_dim=args.embed_dim, decoder=args.decoder, dec_layers=args.dec_layers,
        dec_dim=args.dec_dim, num_heads=args.num_heads, ff_dim=args.ff_dim,
        dropout=args.dropout, rnn_dropout=args.rnn_dropout, max_len=args.max_len,
        level_fusion=args.level_fusion, level_dropout=args.level_dropout,
        height_pool=args.height_pool, fpn_upsample=args.fpn_upsample,
    )
    if args.ckpt and args.eval_only:
        # rebuild the architecture the checkpoint was trained with, if recorded
        cfg_file = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "model_config.json")
        if os.path.isfile(cfg_file):
            model_cfg = json.load(open(cfg_file))
            logger.info(f"model options from {cfg_file}")
        pre_file = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "preprocess.json")
        if os.path.isfile(pre_file):
            pre = json.load(open(pre_file))
            args.keep_ratio = pre.get("keep_ratio", False)
            args.in_chans = pre.get("in_chans", 1)
            logger.info(f"preprocessing from {pre_file}: keep_ratio={args.keep_ratio}")
    model = QACSPTextNet(**model_cfg).to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    if args.init:
        src = torch.load(args.init, map_location=device)
        own = model.state_dict()
        import re

        skip = re.compile(args.init_skip) if args.init_skip else None
        ok = {k: v for k, v in src.items()
              if k in own and own[k].shape == v.shape and not (skip and skip.search(k))}
        for k, v in src.items():  # grayscale stem -> colour stem: copy the filter per channel / 3
            if (k in own and k not in ok and v.ndim == 4 and v.shape[1] == 1
                    and own[k].shape[1] == 3 and own[k].shape[0] == v.shape[0]):
                ok[k] = v.repeat(1, 3, 1, 1) / 3.0
                logger.info(f"--init: {k} converted from 1 to 3 input channels")
        model.load_state_dict(ok, strict=False)
        logger.info(f"--init {args.init}: loaded {len(ok)}/{len(own)} tensors; "
                    f"not loaded: {sorted(set(own) - set(ok))}")

    def eval_loader(src):
        # evaluation keeps every sample (no length / charset filter on GT)
        ds = build_dataset(args.root, src, alphabet=None, img_size=model.img_size)
        for d in getattr(ds, "datasets", [ds]):
            d.keep_ratio = args.keep_ratio
            d.color = args.in_chans == 3
        logger.info(f"{src}: {len(ds)} samples")
        return DataLoader(ds, args.batch_size, num_workers=args.workers, collate_fn=collate)

    val_loaders = {v: eval_loader(v) for v in args.val}
    test_loaders = {t: eval_loader(t) for t in args.test}

    def final_eval():
        accs = {}
        for name, loader in {**val_loaders, **test_loaders}.items():
            accs[name] = evaluate(model, loader, device, args.beam_size)
            logger.info(f"{name}: word accuracy {accs[name]:.4f}")
        if args.metrics_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
            with open(args.metrics_out, "w") as f:
                json.dump({"variant": args.variant, "seed": args.seed, "ckpt": args.ckpt,
                           **{os.path.splitext(os.path.basename(k))[0]: v
                              for k, v in accs.items()}}, f, indent=2)

    if args.eval_only:
        final_eval()
        return

    out_dir = args.output or f"textnet_{args.variant}"
    os.makedirs(out_dir, exist_ok=True)
    logger.add(os.path.join(out_dir, "train_log.txt"))
    logger.info(f"args: {vars(args)}")
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=1)
    with open(os.path.join(out_dir, "preprocess.json"), "w") as f:
        json.dump({"keep_ratio": args.keep_ratio, "aug_type": args.aug_type,
                   "in_chans": args.in_chans, "aug_bend": args.aug_bend,
                   "aug_symbols": args.aug_symbols, "aug_blur": args.aug_blur,
                   "aug_erase": args.aug_erase, "aug_edge": args.aug_edge,
                   "aug_lines": args.aug_lines, "aug_geom": args.aug_geom,
                   "aug_tone": args.aug_tone}, f)
    from torch.utils.data import ConcatDataset

    train_sets = [build_dataset(args.root, t, model.alphabet, args.max_len, model.img_size)
                  for t in args.train]
    if args.augment:
        for ds in train_sets:
            for d in getattr(ds, "datasets", [ds]):
                d.transform = ExploreAugment() if args.aug_type == "explore" else Augment()
                if max(args.aug_bend, args.aug_symbols, args.aug_blur, args.aug_erase,
                       args.aug_edge, args.aug_lines, args.aug_geom, args.aug_tone) > 0:
                    d.pre_transform = CurveSymbolAugment(
                        args.aug_bend, args.aug_symbols, p_blur=args.aug_blur,
                        p_erase=args.aug_erase, p_edge=args.aug_edge, p_lines=args.aug_lines,
                        p_geom=args.aug_geom, p_tone=args.aug_tone)
                    if isinstance(d, RenderDataset):  # rendered words are already degraded
                        d.pre_transform = CurveSymbolAugment(
                            0.4, args.aug_symbols * 0.5, p_blur=0.0,
                            p_erase=0.06, p_edge=args.aug_edge,
                            p_lines=args.aug_lines, p_geom=args.aug_geom, p_tone=args.aug_tone)
                        d.pre_transform.max_ops = 2
                        d.pre_transform.max_span = 120  # readable arcs only
                        d.pre_transform.arc_share = 0.85  # rendered words: mostly true arc writing
                        d.pre_transform.partial_share = 0.35  # of bends: one part of a circle
                        if args.aug_type == "explore":
                            d.transform = ExploreAugment(p_scale=0.3)
    for ds in train_sets:
        for d in getattr(ds, "datasets", [ds]):
            d.keep_ratio = args.keep_ratio
            d.color = args.in_chans == 3
    for t, d in zip(args.train, train_sets):
        logger.info(f"train source {t}: {len(d)} samples")
    sampler = None
    if args.train_weights:
        assert len(args.train_weights) == len(train_sets), "one --train-weights per --train"
        from torch.utils.data import WeightedRandomSampler

        w = torch.cat([torch.full((len(d),), wt / len(d), dtype=torch.double)
                       for d, wt in zip(train_sets, args.train_weights)])
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        tot = sum(args.train_weights)
        for t, wt in zip(args.train, args.train_weights):
            logger.info(f"batch share {t}: {wt / tot:.3f}")
    train_loader = DataLoader(
        ConcatDataset(train_sets), args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=args.workers, collate_fn=collate, drop_last=True,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    if args.freeze_backbone:
        # freeze the CSPDarknet backbone (stem, dark2-5): no gradients, BatchNorm statistics fixed
        bb = model.backbone.backbone
        for p_ in bb.parameters():
            p_.requires_grad_(False)
        model._frozen_bb = True  # evaluate() puts the backbone back in eval mode after model.train()
        bb.eval()  # BN running stats stay as loaded
        n_tr = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
        logger.info(f"--freeze-backbone: backbone frozen ({sum(p_.numel() for p_ in bb.parameters())/1e6:.2f}M); "
                    f"trainable {n_tr/1e6:.2f}M (neck, AFW, fusion, fc_in, BLSTM, CTC)")
    optimizer = build_optimizer([p_ for p_ in model.parameters() if p_.requires_grad], args)
    total_for_sched = args.iters or args.epochs * len(train_loader)
    if args.time_limit and args.iters:
        logger.info("the lr schedule follows --iters; set --iters to what fits in --time-limit")

    def lr_at(step):
        return lr_factor(step, total_for_sched, args) * state["plateau"] * args.lr
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    from qacspnet.utils import ModelEMA

    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    state = {"best": -1.0, "step": args.start_step, "plateau": 1.0, "bad_evals": 0,
             "last": -1.0, "stop": False}
    if args.stop_window:
        lo, hi = args.stop_window
        if args.keep_from is None:
            args.keep_from = lo - 1.0
        if args.fine_eval_from is None:
            args.fine_eval_from = lo - 1.5

    def eval_interval():
        # evaluate more often near the target so the window is not skipped
        if args.fine_eval_from is not None and state["last"] * 100 >= args.fine_eval_from:
            return min(args.fine_eval_interval, args.eval_interval)
        return args.eval_interval
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state.update(best=ck["best"], step=ck["step"], plateau=ck.get("plateau", 1.0),
                     bad_evals=ck.get("bad_evals", 0))
        if ema is not None and ck.get("ema") is not None:
            ema.ema.load_state_dict(ck["ema"])
            ema.updates = ck.get("ema_updates", 0)
        logger.info(f"resumed from {args.resume} at step {state['step']}")
    elif args.start_step:
        logger.info(f"continuing from --ckpt {args.ckpt} at step {args.start_step} "
                    f"(optimizer state not restored)")
    if os.path.isfile(os.path.join(out_dir, "best_score.txt")):
        # last_state is written before its evaluation, so best_score.txt can be newer
        state["best"] = max(state["best"],
                            float(open(os.path.join(out_dir, "best_score.txt")).read()))

    def save_state():
        """Resume point: model, optimizer, scaler, step, EMA (written atomically)."""
        net = ema.ema if ema is not None else model
        save_atomic(net.state_dict(), os.path.join(out_dir, "last.pth"))
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": state["step"], "best": state["best"],
                    "plateau": state["plateau"], "bad_evals": state["bad_evals"],
                    "ema": ema.ema.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0},
                   os.path.join(out_dir, "last_state.pth"))

    def checkpoint(tag):
        net = ema.ema if ema is not None else model  # weights that are evaluated and saved
        save_state()
        accs = {n: evaluate(net, l, device, args.beam_size) for n, l in val_loaders.items()}
        for n, a in accs.items():
            logger.info(f"{tag} {n}: word accuracy {a:.4f}")
        score = sum(accs.values()) / max(len(accs), 1)
        if accs:
            state["last"] = score
            pct = 100 * score
            if args.keep_from is not None and pct >= args.keep_from:
                save_atomic(net.state_dict(), os.path.join(
                    out_dir, f"ckpt_iter{state['step']}_{pct:.2f}.pth"))
            if args.stop_window and args.stop_window[0] <= pct <= args.stop_window[1]:
                save_atomic(net.state_dict(), os.path.join(out_dir, "target.pth"))
                with open(os.path.join(out_dir, "target.txt"), "w") as f:
                    f.write(f"step {state['step']} accuracy {pct:.2f}\n")
                logger.info(f"{tag}: accuracy {pct:.2f} in stop window "
                            f"{args.stop_window}: saved target.pth, stopping")
                state["stop"] = True
        if accs and score > state["best"]:
            state["best"] = score
            save_atomic(net.state_dict(), os.path.join(out_dir, "best.pth"))
            with open(os.path.join(out_dir, "best_score.txt"), "w") as f:
                f.write(str(score))
            logger.info(f"{tag}: new best {score:.4f}")
            state["bad_evals"] = 0
        elif accs and args.sched == "plateau":
            state["bad_evals"] += 1
            if state["bad_evals"] >= args.patience:
                state["plateau"] *= args.gamma
                state["bad_evals"] = 0
                logger.info(f"{tag}: no improvement for {args.patience} evaluations, "
                            f"lr scaled by {args.gamma}")

    total = args.iters or args.epochs * len(train_loader)
    import time
    t_start = time.time()

    def out_of_time():
        return args.time_limit and time.time() - t_start > args.time_limit * 3600

    epoch = 0
    while state["step"] < total and not out_of_time() and not state["stop"]:
        for imgs, texts in train_loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                loss = model.compute_loss(imgs.to(device, non_blocking=True), texts)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(state["step"])
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            state["step"] += 1
            step = state["step"]
            if step % 100 == 0:
                logger.info(f"epoch {epoch} iter {step}/{total} loss {loss.item():.4f} "
                            f"lr {optimizer.param_groups[0]['lr']:.2e}")
            if args.iters and step % eval_interval() == 0:
                checkpoint(f"iter {step}")
            elif args.save_interval and step % args.save_interval == 0:
                save_state()  # frequent resume point (survives reboots)
            if step >= total or out_of_time() or state["stop"]:
                if out_of_time():
                    logger.info(f"time limit reached at iter {step}")
                break
        if not args.iters:
            checkpoint(f"epoch {epoch}")
        epoch += 1
    if not state["stop"] and ((args.iters and state["step"] % eval_interval()) or out_of_time()):
        checkpoint(f"iter {state['step']}")

    # final evaluation with the checkpoint selected on the dev files
    best_path = os.path.join(out_dir, "best.pth" if val_loaders else "last.pth")
    if state["stop"]:
        best_path = os.path.join(out_dir, "target.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))
    args.ckpt = best_path
    final_eval()


if __name__ == "__main__":
    main()
