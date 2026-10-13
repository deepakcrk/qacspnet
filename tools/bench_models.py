"""Speed / size benchmark of trained QA-CSPTextNet models (user 2026-10-11).
   python tools/bench_models.py OUT.md name=ckpt_dir [name=ckpt_dir ...]
Per model (its own model_config.json, img size from it, FP32, eval mode, torch.inference_mode):
  params; FLOPs per image (torch FlopCounterMode, + analytic BLSTM if not counted; GFLOPs = 2 x GMACs);
  CPU latency batch 1; GPU latency/throughput batch 1, 8, 16, 32 (cudnn.benchmark, 20 warm-up, 100 timed,
  median; end-to-end = forward + greedy CTC decode)."""
import sys, json, time, os, statistics as st, torch
from torch.utils.flop_counter import FlopCounterMode
from qacspnet.models import QACSPTextNet

VARIANTS = {"s": (0.33, 0.5), "m": (0.67, 0.5), "l": (1.0, 0.5), "x": (1.33, 0.5)}

def load(d):
    """d = run dir (trained best.pth), or rand:<cfg_run_dir>:<variant>:<H>x<W> (random init, same
    architecture settings as that run's model_config; speed/FLOPs/params do not depend on weights)."""
    if d.startswith("rand:"):
        _, base, v, hw = d.split(":"); cfg = json.load(open(f"{base}/model_config.json"))
        cfg["depth"], cfg["width"] = VARIANTS[v]; cfg["img_size"] = [int(t) for t in hw.split("x")]
        return QACSPTextNet(**cfg).eval(), cfg
    cfg = json.load(open(f"{d}/model_config.json")); m = QACSPTextNet(**cfg)
    m.load_state_dict(torch.load(f"{d}/best.pth", map_location="cpu")); return m.eval(), cfg

def flops(m, x):
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        m(x)
    total = fc.get_total_flops(); per = fc.get_flop_counts()
    rnn_counted = any("rnn" in k.lower() or "lstm" in k.lower() for k in per if sum(per[k].values()) > 0)
    lstm = 0
    if not rnn_counted:
        for mod in m.modules():
            if isinstance(mod, torch.nn.LSTM):
                T, h = x.shape[-1] // 4, mod.hidden_size; dirs = 2 if mod.bidirectional else 1
                for l in range(mod.num_layers):
                    i = mod.input_size if l == 0 else h * dirs
                    lstm += dirs * T * 2 * 4 * (i * h + h * h)
    return (total + lstm) / x.shape[0], lstm / x.shape[0]

def timeit(m, x, dev, n_warm, n, forward_only=False):
    ts = []
    with torch.inference_mode():
        for k in range(n_warm + n):
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); (m(x) if forward_only else m.recognize(x))
            if dev == "cuda": torch.cuda.synchronize()
            if k >= n_warm: ts.append(time.perf_counter() - t0)
    return st.median(ts)

def paper_protocol(out, specs):
    """Table 9 protocol of the paper: forward pass only (no CTC decoding), GPU FP16 (.half()) / FP32 at
    batch 1, 16, 32 (30 warm-up, 200 timed, median), CPU FP32 batch 1 (10 warm-up, 50 timed)."""
    torch.backends.cudnn.benchmark = True; rows = []
    for name, d in specs:
        m, cfg = load(d); H, W = cfg["img_size"]; c = cfg.get("in_chans", 1)
        r = {"model": name, "ckpt": d if d.startswith("rand:") else f"{d}/best.pth", "input": f"{H}x{W}",
             "params_M": sum(p.numel() for p in m.parameters()) / 1e6, "GFLOPs": flops(m, torch.zeros(1, c, H, W))[0] / 1e9}
        r["cpu_fp32_b1_ms"] = 1000 * timeit(m, torch.zeros(1, c, H, W), "cpu", 10, 50, True)
        for prec in ("fp32", "fp16"):
            mg = m.cuda().half() if prec == "fp16" else m.cuda().float()
            for b in (1, 16, 32):
                x = torch.zeros(b, c, H, W, device="cuda", dtype=torch.float16 if prec == "fp16" else torch.float32)
                r[f"gpu_{prec}_b{b}_ms_per_img"] = 1000 * timeit(mg, x, "cuda", 30, 200, True) / b
            m.float().cpu(); torch.cuda.empty_cache()
        rows.append(r); print(json.dumps(r), flush=True)
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0),
            "cpu": os.popen("lscpu | grep 'Model name' | cut -d: -f2").read().strip(), "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__, "protocol": "paper Table 9: forward only, no CTC decode; GPU fp16 (.half) / fp32, "
            "batch 1/16/32, 30 warm-up + 200 timed, median per image; CPU fp32 batch 1, 10 warm-up + 50 timed"}
    json.dump({"meta": meta, "models": rows}, open(out.replace(".md", ".json"), "w"), indent=1)
    L = [f"# Table 9 protocol benchmark ({time.strftime('%a %b %d, %I:%M %p')})", "", meta["protocol"] + f". GPU {meta['gpu']}; CPU {meta['cpu']}, {meta['cpu_threads']} threads.", "",
         "| Model | Input | Params (M) | GFLOPs | GPU b=1 FP16 / FP32 | GPU b=16 FP16 / FP32 | GPU b=32 FP16 / FP32 | CPU b=1 FP32 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['params_M']:.2f} | {r['GFLOPs']:.2f} | "
                 + " | ".join(f"{r[f'gpu_fp16_b{b}_ms_per_img']:.2f} / {r[f'gpu_fp32_b{b}_ms_per_img']:.2f}" for b in (1, 16, 32))
                 + f" | {r['cpu_fp32_b1_ms']:.1f} |")
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    if sys.argv[1] == "--paper":
        return paper_protocol(sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]])
    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    torch.backends.cudnn.benchmark = True
    rows = []
    for name, d in specs:
        m, cfg = load(d); H, W = cfg["img_size"]; c = cfg.get("in_chans", 1)
        params = sum(p.numel() for p in m.parameters())
        f, fl = flops(m, torch.zeros(1, c, H, W))
        r = {"model": name, "ckpt": d if d.startswith("rand:") else f"{d}/best.pth", "input": f"{H}x{W}", "params_M": params / 1e6,
             "GFLOPs": f / 1e9, "GMACs": f / 2e9, "lstm_GFLOPs_added": fl / 1e9}
        r["cpu_b1_ms"] = 1000 * timeit(m, torch.zeros(1, c, H, W), "cpu", 5, 30)
        mg = m.cuda()
        for b in (1, 8, 16, 32):
            t = timeit(mg, torch.zeros(b, c, H, W, device="cuda"), "cuda", 20, 100)
            r[f"gpu_b{b}_ms"] = 1000 * t; r[f"gpu_b{b}_img_s"] = b / t
        m.cpu(); del mg; torch.cuda.empty_cache()
        rows.append(r); print(json.dumps(r), flush=True)
    gpu = torch.cuda.get_device_name(0); cpu = os.popen("lscpu | grep 'Model name' | cut -d: -f2").read().strip()
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": gpu, "cpu": cpu, "cpu_threads": torch.get_num_threads(),
            "torch": torch.__version__, "precision": "fp32", "mode": "eval, torch.inference_mode, end-to-end forward + greedy CTC",
            "timing": "median; CPU 5 warm-up + 30 timed (batch 1); GPU 20 warm-up + 100 timed (batch 1/8/16/32), cudnn.benchmark",
            "flops": "torch FlopCounterMode + analytic BLSTM; GFLOPs = 2 x GMACs, per image"}
    json.dump({"meta": meta, "models": rows}, open(out.replace(".md", ".json"), "w"), indent=1)
    L = [f"# Speed / size benchmark ({time.strftime('%a %b %d, %I:%M %p')})", "",
         f"FP32, eval mode, end-to-end (forward + greedy CTC). GPU: {gpu}. CPU: {cpu}, {torch.get_num_threads()} threads.",
         "Latency = median of timed runs (CPU 30, GPU 100) after warm-up. GFLOPs = 2 x GMACs per image.", "",
         "| Model | Input | Params (M) | GFLOPs / img | GMACs / img | CPU b=1 (ms) | GPU b=1 (ms) | GPU b=8 (ms) | GPU b=16 (ms) | GPU b=32 (ms) | GPU b=32 (img/s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['params_M']:.2f} | {r['GFLOPs']:.2f} | {r['GMACs']:.2f} | {r['cpu_b1_ms']:.1f} | "
                 f"{r['gpu_b1_ms']:.2f} | {r['gpu_b8_ms']:.2f} | {r['gpu_b16_ms']:.2f} | {r['gpu_b32_ms']:.2f} | {r['gpu_b32_img_s']:.0f} |")
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main()
