#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# QA-CSPTextNet: QA-CSPNet backbone + sequence encoder + decoder for scene text
# recognition. Default: 2 x BLSTM(256) + CTC.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qacsp_pafpn import QACSPPAFPN

DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


_UPSAMPLE_CACHE = {}


def _linear_upsample_matrix(w_in, w_out, device, dtype):
    """w_in x w_out matrix M with x @ M == linear interpolation (align_corners=False)."""
    key = (w_in, w_out, device, dtype)
    if key not in _UPSAMPLE_CACHE:
        eye = torch.eye(w_in, device=device, dtype=torch.float32)[None]  # 1 x w_in x w_in
        m = F.interpolate(eye, size=w_out, mode="linear", align_corners=False)[0]
        _UPSAMPLE_CACHE[key] = m.to(dtype)
    return _UPSAMPLE_CACHE[key]


class QACSPTextNet(nn.Module):
    """Input B x 1 x H x W (grayscale; default 32 x 96, H and W multiples of 16).
    The three PAFPN levels (strides 4, 8, 16; for 32 x 96 and width=0.5:
    128x8x24, 256x4x12, 512x2x6) are each reshaped to (width, B, C*height) and
    concatenated along the width into a T x B x D sequence (T = W/4 + W/8 + W/16,
    42 for W = 96) -> FC(embed_dim) -> encoder -> decoder.

    encoder:
      "blstm"        enc_layers x BLSTM, `hidden` units per direction (default),
                     `rnn_dropout` between layers
      "transformer"  enc_layers x Transformer encoder layer (self-attention),
                     model size embed_dim, `num_heads` heads, `ff_dim` feed-forward
      "none"         no sequence encoder
    decoder:
      "ctc"   linear classifier + CTC (default); class 0 is the CTC blank
      "attn"  dec_layers x Transformer decoder layer (attention decoder) with
              cross-attention to the encoder output, trained with cross-entropy
              and decoded greedily (max_len characters); model size dec_dim
    `dropout` applies to the Transformer encoder and decoder layers.
    """

    def __init__(
        self,
        depth=0.33,
        width=0.5,
        img_size=(32, 96),
        in_chans=1,
        alphabet=DEFAULT_ALPHABET,
        embed_dim=64,
        hidden=256,
        act="silu",
        use_quality=True,
        use_afw=True,
        afw_cfg=None,
        qge_cfg=None,
        gce_cfg=None,
        encoder="blstm",
        enc_layers=2,
        decoder="ctc",
        dec_layers=2,
        dec_dim=256,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
        rnn_dropout=0.0,
        max_len=25,
        level_fusion="time",
        level_dropout=0.0,
        height_pool="none",
        fpn_upsample="bilinear",
    ):
        super().__init__()
        assert encoder in ("blstm", "transformer", "none"), encoder
        assert decoder in ("ctc", "attn"), decoder
        # how the three PAFPN levels (strides 4, 8, 16) form the sequence:
        #   "time"    levels concatenated along the time axis (W/4 + W/8 + W/16 steps; original)
        #   "channel" strides 8 and 16 linearly upsampled along the width to W/4 and
        #             concatenated per position (W/4 steps, each with all three levels)
        #   "fpn"     per level 1x1 conv + BN + SiLU to the stride-4 channel count, upsampled
        #             (height and width) to the stride-4 map, appended, 1x1 conv + BN + SiLU
        #             -> W/4 steps (stride-4 shape)
        assert level_fusion in ("time", "channel", "fpn"), level_fusion
        self.level_fusion = level_fusion
        # training only (channel fusion): for this fraction of images one randomly chosen level
        # is zeroed, so the model cannot rely on a single level
        self.level_dropout = level_dropout
        # height -> 1 per column: "none" flattens C x H into the step vector (default),
        # "avg" / "max" pool over the height (vertical-shift invariant, step vector = C)
        # "conv": CRNN-style learned collapse, 3x1 convs with vertical stride 2 until the height
        # is <= 3, then an (H x 1) conv to height 1 (BN + SiLU after each); only with fpn fusion
        assert height_pool in ("none", "avg", "max", "conv"), height_pool
        self.height_pool = height_pool
        # how strides 8 / 16 reach the stride-4 grid in the fpn block: fixed bilinear, or learned
        # (1x1 conv expanding channels + pixel shuffle, or a transposed conv)
        assert fpn_upsample in ("bilinear", "shuffle", "deconv"), fpn_upsample
        self.fpn_upsample = fpn_upsample
        assert len(set(alphabet)) == len(alphabet), "alphabet has repeated characters"
        self.alphabet = alphabet
        # labels are lower-cased unless the alphabet itself contains upper-case letters
        self.case_sensitive = any(ch != ch.lower() for ch in alphabet)
        self.encoder_type, self.decoder_type = encoder, decoder
        self.max_len = max_len
        self.backbone = QACSPPAFPN(
            depth, width, in_chans=in_chans, use_focus=False, act=act,
            use_quality=use_quality, use_afw=use_afw, afw_cfg=afw_cfg, qge_cfg=qge_cfg,
            gce_cfg=gce_cfg,
        )
        # feature size per time step for each level (C * H)
        h, w = img_size
        if h % 16 or w % 16:
            raise ValueError(f"img_size {img_size}: height and width must be multiples of 16")
        self.img_size = (h, w)
        self.seq_len = w // 4 if level_fusion in ("channel", "fpn") else w // 4 + w // 8 + w // 16
        feat_dims = [
            int(c * width) * (h // s)
            for c, s in zip(self.backbone.in_channels, (4, 8, 16))
        ]
        assert len(set(feat_dims)) == 1, f"levels give different C*H: {feat_dims}"
        n_in = feat_dims[0] * (3 if level_fusion == "channel" else 1)
        if height_pool in ("avg", "max"):
            n_in //= (h // 4)  # pooled: channels only (C instead of C * H per level)
        if level_fusion == "fpn":
            chans = [int(c * width) for c in self.backbone.in_channels]
            c0 = chans[0]

            def cbs(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.SiLU())

            if fpn_upsample == "shuffle":  # 1x1 conv to c0 * r^2 channels, then pixel shuffle (r)
                self.fuse_lat = nn.ModuleList([cbs(c, c0 * r * r) for c, r in zip(chans[1:], (2, 4))])
            elif fpn_upsample == "deconv":  # transposed conv, kernel = stride = r
                self.fuse_lat = nn.ModuleList([
                    nn.Sequential(nn.ConvTranspose2d(c, c0, r, stride=r, bias=False),
                                  nn.BatchNorm2d(c0), nn.SiLU())
                    for c, r in zip(chans[1:], (2, 4))])
            else:
                self.fuse_lat = nn.ModuleList([cbs(c, c0) for c in chans[1:]])  # strides 8, 16
            self.fuse_out = cbs(3 * c0, c0)
            # residual fusion: output = stride-4 map + fuse_out(...); the last BN starts at
            # zero so the block initially passes the stride-4 map through unchanged
            nn.init.zeros_(self.fuse_out[1].weight)
        if height_pool == "conv":
            assert level_fusion == "fpn", "--height-pool conv needs --level-fusion fpn"
            hcur, layers, cout = h // 4, [], 256
            while hcur > 3:
                layers += [nn.Conv2d(c0, c0, (3, 1), (2, 1), (1, 0), bias=False), nn.BatchNorm2d(c0), nn.SiLU()]
                hcur = (hcur + 1) // 2
            layers += [nn.Conv2d(c0, cout, (hcur, 1), bias=False), nn.BatchNorm2d(cout), nn.SiLU()]
            self.hcollapse = nn.Sequential(*layers)
            n_in = cout
        self.fc_in = nn.Linear(n_in, embed_dim)

        # ---- sequence encoder ----
        if encoder == "blstm":
            self.rnn = nn.LSTM(embed_dim, hidden, num_layers=enc_layers, bidirectional=True,
                               dropout=rnn_dropout if enc_layers > 1 else 0.0)
            enc_out = 2 * hidden
        elif encoder == "transformer":
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            self.enc_pos = nn.Parameter(torch.zeros(self.seq_len, 1, embed_dim))
            nn.init.trunc_normal_(self.enc_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                               activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
            enc_out = embed_dim
        else:
            enc_out = embed_dim

        # ---- decoder ----
        if decoder == "ctc":
            self.num_classes = len(alphabet) + 1  # blank + characters
            self.fc_out = nn.Linear(enc_out, self.num_classes)
        else:
            # tokens: 0 = pad, 1..K = characters, K+1 = end, K+2 = start
            assert dec_dim % num_heads == 0, "dec_dim must be divisible by num_heads"
            self.pad_id, self.eos_id, self.sos_id = 0, len(alphabet) + 1, len(alphabet) + 2
            self.num_classes = len(alphabet) + 2  # pad + characters + end
            self.mem_proj = nn.Linear(enc_out, dec_dim)
            self.tok_emb = nn.Embedding(len(alphabet) + 3, dec_dim)
            self.dec_pos = nn.Parameter(torch.zeros(max_len + 2, 1, dec_dim))
            nn.init.trunc_normal_(self.dec_pos, std=0.02)
            layer = nn.TransformerDecoderLayer(dec_dim, num_heads, ff_dim, dropout,
                                               activation="gelu")
            self.attn_decoder = nn.TransformerDecoder(layer, dec_layers)
            self.fc_out = nn.Linear(dec_dim, self.num_classes)

    # ------------------------------------------------------------------ #
    def _collapse(self, f):
        """B x C x H x W -> B x (C*H) x W (flatten) or B x C x W (height pooling)."""
        b, c, h, w = f.shape
        if self.height_pool == "conv":
            return self.hcollapse(f).squeeze(2)  # B x 256 x W
        if self.height_pool == "avg":
            return f.mean(dim=2)
        if self.height_pool == "max":
            return f.amax(dim=2)
        return f.reshape(b, c * h, w)

    def encode_features(self, x):
        """Image -> encoder output, T x B x D."""
        feats = self.backbone(x)  # fine -> coarse
        if self.level_fusion == "fpn":
            f0 = feats[0]
            h0, w0 = f0.shape[-2:]
            maps = [f0]
            for lat, f in zip(self.fuse_lat, feats[1:]):
                f = lat(f)  # B x c0 x h x w (bilinear) / B x c0*r^2 x h x w (shuffle) / upsampled (deconv)
                if self.fpn_upsample == "shuffle":
                    f = F.pixel_shuffle(f, h0 // f.shape[-2])
                elif self.fpn_upsample == "bilinear":
                    # separable bilinear upsampling as two matmuls (= F.interpolate bilinear,
                    # align_corners=False; the interpolate backward kernel is slow)
                    f = f @ _linear_upsample_matrix(f.shape[-1], w0, f.device, f.dtype)
                    f = (f.transpose(-1, -2) @ _linear_upsample_matrix(
                        f.shape[-2], h0, f.device, f.dtype)).transpose(-1, -2)
                maps.append(f)
            if self.training and self.level_dropout > 0:
                b = f0.shape[0]
                drop = torch.rand(b, device=x.device) < self.level_dropout
                which = torch.randint(0, len(maps), (b,), device=x.device)
                maps = [m_ * (~(drop & (which == i))).to(m_.dtype)[:, None, None, None]
                        for i, m_ in enumerate(maps)]
            f = maps[0] + self.fuse_out(torch.cat(maps, dim=1))  # B x c0 x h0 x w0
            b, c, h, w = f.shape
            col = self._collapse(f)  # B x (C*H or C) x W0
            seq = self.fc_in(col.permute(2, 0, 1))  # W0 x B x embed_dim
        elif self.level_fusion == "channel":
            w0 = feats[0].shape[-1]
            cols = []
            for f in feats:
                b, c, h, w = f.shape
                f = self._collapse(f)  # B x CH (or C) x W
                if w != w0:
                    # = F.interpolate(f, size=w0, mode="linear", align_corners=False),
                    # done as a matmul (the interpolate backward kernel is very slow)
                    f = f @ _linear_upsample_matrix(w, w0, f.device, f.dtype)
                cols.append(f)
            if self.training and self.level_dropout > 0:
                b = cols[0].shape[0]
                drop = torch.rand(b, device=x.device) < self.level_dropout
                which = torch.randint(0, len(cols), (b,), device=x.device)
                cols = [c * (~(drop & (which == i))).to(c.dtype)[:, None, None]
                        for i, c in enumerate(cols)]
            seq = self.fc_in(torch.cat(cols, dim=1).permute(2, 0, 1))  # W0 x B x embed_dim
        else:
            seq = []
            for f in feats:
                b, c, h, w = f.shape
                seq.append(f.reshape(b, c * h, w).permute(2, 0, 1))  # W x B x CH
            seq = self.fc_in(torch.cat(seq, dim=0))  # T x B x embed_dim
        if self.encoder_type == "blstm":
            seq, _ = self.rnn(seq)
        elif self.encoder_type == "transformer":
            seq = self.transformer(seq + self.enc_pos[: seq.size(0)])
        return seq

    def forward(self, x):
        """CTC: T x B x C log-probs. Attention decoder: greedy token ids B x L."""
        mem = self.encode_features(x)
        if self.decoder_type == "ctc":
            return self.fc_out(mem).log_softmax(-1)
        return self._greedy(mem)

    # ---------------- attention decoder ---------------- #
    def _decode_step(self, tokens, mem):
        """tokens L x B (starting with SOS) -> logits L x B x num_classes."""
        L = tokens.size(0)
        tgt = self.tok_emb(tokens) + self.dec_pos[:L]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), 1)
        out = self.attn_decoder(tgt, mem, tgt_mask=mask)
        return self.fc_out(out)

    @torch.no_grad()
    def _greedy(self, mem):
        mem = self.mem_proj(mem)
        B = mem.size(1)
        tokens = torch.full((1, B), self.sos_id, dtype=torch.long, device=mem.device)
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        for _ in range(self.max_len + 1):
            nxt = self._decode_step(tokens, mem)[-1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            tokens = torch.cat([tokens, nxt[None]], 0)
            done |= nxt == self.eos_id
            if done.all():
                break
        return tokens[1:].t()  # B x L

    # ---------------- training / inference API ---------------- #
    def _char_ids(self, text):
        idx = {ch: i + 1 for i, ch in enumerate(self.alphabet)}
        return [idx[ch] for ch in self.normalize(text) if ch in idx]

    def normalize(self, text):
        """Label as the model sees it: lower-cased unless case-sensitive."""
        text = text.strip()
        return text if self.case_sensitive else text.lower()

    def compute_loss(self, imgs, texts):
        """Recognition loss for a batch of images and their transcriptions."""
        mem = self.encode_features(imgs)
        if self.decoder_type == "ctc":
            targets, target_lens = self.encode(texts)
            log_probs = self.fc_out(mem).float().log_softmax(-1)  # CTC in fp32
            input_lens = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            return F.ctc_loss(log_probs, targets, input_lens, target_lens,
                              blank=0, zero_infinity=True)
        ids = [self._char_ids(t)[: self.max_len] + [self.eos_id] for t in texts]
        L = max(len(i) for i in ids)
        tgt_out = torch.full((L, len(ids)), self.pad_id, dtype=torch.long)
        for b, i in enumerate(ids):
            tgt_out[: len(i), b] = torch.tensor(i)
        tgt_out = tgt_out.to(imgs.device)
        sos = torch.full((1, len(ids)), self.sos_id, dtype=torch.long, device=imgs.device)
        tgt_in = torch.cat([sos, tgt_out[:-1]], 0)
        logits = self._decode_step(tgt_in, self.mem_proj(mem)).float()
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                               ignore_index=self.pad_id)

    @torch.no_grad()
    def recognize(self, imgs, beam_size=1):
        """Images -> list of predicted words (beam_size > 1: CTC beam search)."""
        return self.decode(self(imgs), beam_size)

    def encode(self, texts):
        """list[str] -> (flat targets, lengths) for CTC."""
        targets, lengths = [], []
        for t in texts:
            ids = self._char_ids(t)
            targets.extend(ids)
            lengths.append(len(ids))
        return torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    @torch.no_grad()
    def decode(self, out, beam_size=1):
        """CTC log-probs (T x B x C) or attention token ids (B x L) -> list[str]."""
        if self.decoder_type == "ctc" and beam_size > 1:
            lp = out.float().permute(1, 0, 2).cpu()
            return [self._ctc_beam(seq, beam_size) for seq in lp]
        if self.decoder_type == "attn":
            words = []
            for seq in out.cpu().tolist():
                chars = []
                for i in seq:
                    if i == self.eos_id:
                        break
                    if 1 <= i <= len(self.alphabet):
                        chars.append(self.alphabet[i - 1])
                words.append("".join(chars))
            return words
        best = out.argmax(-1).permute(1, 0).cpu().tolist()
        words = []
        for seq in best:
            chars, prev = [], 0
            for i in seq:
                if i != prev and i != 0:
                    chars.append(self.alphabet[i - 1])
                prev = i
            words.append("".join(chars))
        return words

    def _ctc_beam(self, log_probs, beam_size):
        """CTC prefix beam search over one T x C log-prob sequence (blank = 0)."""
        neg_inf = float("-inf")

        def lse(a, b):
            if a == neg_inf:
                return b
            if b == neg_inf:
                return a
            m = max(a, b)
            return m + math.log(math.exp(a - m) + math.exp(b - m))

        beams = {(): (0.0, neg_inf)}  # prefix -> (log p ending in blank, ending in non-blank)
        topk = min(beam_size, log_probs.size(1))
        for t in range(log_probs.size(0)):
            row = log_probs[t]
            cands = set(row.topk(topk).indices.tolist()) | {0}
            nxt = {}
            for prefix, (pb, pnb) in beams.items():
                for c in cands:
                    p = row[c].item()
                    if c == 0:
                        b, nb = nxt.get(prefix, (neg_inf, neg_inf))
                        nxt[prefix] = (lse(b, lse(pb, pnb) + p), nb)
                        continue
                    new = prefix + (c,)
                    b, nb = nxt.get(new, (neg_inf, neg_inf))
                    if prefix and prefix[-1] == c:
                        nxt[new] = (b, lse(nb, pb + p))  # repeat needs a blank in between
                        ob, onb = nxt.get(prefix, (neg_inf, neg_inf))
                        nxt[prefix] = (ob, lse(onb, pnb + p))  # collapse repeat
                    else:
                        nxt[new] = (b, lse(nb, lse(pb, pnb) + p))
            beams = dict(sorted(nxt.items(), key=lambda kv: -lse(*kv[1]))[:beam_size])
        best = max(beams.items(), key=lambda kv: lse(*kv[1]))[0]
        return "".join(self.alphabet[i - 1] for i in best)

