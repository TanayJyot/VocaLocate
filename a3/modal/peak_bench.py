"""
A3 Part 3: measure each GPU's dense bf16 matrix-multiply peak, for the MFU of GPUs missing from nanochat's table.
H100 runs too, so the gap between a measured peak and a rated one is known before the two are compared.

    modal run peak_bench.py
"""
import json
import subprocess
import time

import modal

TORCH_2_9 = modal.Image.debian_slim(python_version="3.10").pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
# B300 needs CUDA 13.1+
TORCH_2_15 = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cu132")
app = modal.App("vocalocate-a3-peak-bench")


def _bench():
    import torch

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    out = {"gpu": gpu, "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "tflops": {}}
    for dtype in (torch.bfloat16, torch.float32):
        name = "bf16" if dtype == torch.bfloat16 else "fp32"
        for n in (2048, 4096, 8192, 16384):
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            torch.matmul(a, b)  # first call sets up kernels
            torch.cuda.synchronize()
            best, spent = 0.0, 0.0
            while spent < 20:
                t0 = time.perf_counter()
                c = torch.matmul(a, b)
                c[0, 0].item()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                spent += dt
                best = max(best, 2 * n**3 / dt / 1e12)
            out["tflops"][f"{name}_{n}"] = round(best, 1)
            del a, b, c
            torch.cuda.empty_cache()
    print("PEAK", json.dumps(out), flush=True)
    return out


@app.function(gpu="H100!", image=TORCH_2_9, timeout=900)
def h100():
    return _bench()


@app.function(gpu="H100!", image=TORCH_2_15, timeout=900)
def h100_new_torch():
    return _bench()


@app.function(gpu="RTX-PRO-6000", image=TORCH_2_9, timeout=900)
def rtx():
    return _bench()


@app.function(gpu="A100-80GB", image=TORCH_2_9, timeout=900)
def a100():
    return _bench()


@app.function(gpu="L40S", image=TORCH_2_9, timeout=900)
def l40s():
    return _bench()


# B200+ is billed as B200, and it may hand out either card; the result names the one it got
@app.function(gpu="B200+", image=TORCH_2_15, timeout=900)
def b300():
    return _bench()


@app.local_entrypoint()
def main(only: str = "h100,h100_new_torch,rtx,b300,a100,l40s"):
    funcs = {"h100": h100, "h100_new_torch": h100_new_torch, "rtx": rtx, "b300": b300, "a100": a100, "l40s": l40s}
    calls = {n: funcs[n].spawn() for n in only.split(",")}
    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get(timeout=1500)
        except Exception as e:
            results[name] = {"error": repr(e)[:500]}
        print(name, json.dumps(results[name]), flush=True)
    with open(f"peak_bench_results_{only.replace(',', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)
