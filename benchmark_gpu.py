"""
benchmark_gpu.py  —  GPU Pipeline

Benchmarks GPU-compatible models (INT8/FP16) using OpenVINO benchmark_app -d GPU.
Scans for: baseline + gpu_point{01-10} + uniform_int8.

Output:
    log/benchmark_log/{model_name}_GPU_latency.txt   — raw benchmark_app output
    log/benchmark_log/latency_throughput_gpu_report.csv — summary CSV

Usage:
    python benchmark_gpu.py
"""

import os
import re
import csv
import shutil
import subprocess

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PREFIXES = [
    "mamba-130m-hf",
    "mamba2_b_1_t_4",
]

OV_MODELS_DIR  = "ov_models"
LOG_DIR        = "log/benchmark_log"
DEVICE         = "GPU"
HINT           = "latency"
DURATION       = 30

LATENCY_RE    = re.compile(r"\[ INFO \]\s+Average:\s+([\d\.]+)\s+ms")
THROUGHPUT_RE = re.compile(r"\[ INFO \]\s+Throughput:\s+([\d\.]+)\s+FPS")

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_gpu_models(models_dir, prefixes):
    """
    Find GPU-compatible .xml files:
      - {prefix}.xml              (baseline)
      - {prefix}_gpu_point*.xml   (mixed INT8/FP16)
      - {prefix}_uniform_int8.xml (uniform INT8)
    """
    results = []
    for f in sorted(os.listdir(models_dir)):
        if not f.endswith(".xml"):
            continue
        if not any(f.startswith(p) for p in prefixes):
            continue

        stem = f[:-4]   # strip .xml
        for prefix in prefixes:
            if not f.startswith(prefix):
                continue
            suffix = stem[len(prefix):]   # everything after model prefix

            # Accept: baseline, _gpu_point*, _uniform_int8
            # (mamba-1.4b baseline was excluded — 5.3GB, 30+ min compile;
            #  mamba-130m and mamba2-130m baselines are fine)
            if suffix == "" or suffix.startswith("_gpu_point") or suffix == "_uniform_int8":
                results.append((f, os.path.join(models_dir, f)))
            break  # matched a prefix, no need to check others

    return results


def parse_log(log_path):
    latency = throughput = None
    try:
        with open(log_path) as f:
            for line in f:
                if latency is None:
                    m = LATENCY_RE.search(line)
                    if m:
                        latency = float(m.group(1))
                if throughput is None:
                    m = THROUGHPUT_RE.search(line)
                    if m:
                        throughput = float(m.group(1))
                if latency and throughput:
                    break
    except FileNotFoundError:
        pass
    return latency, throughput

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Locate benchmark_app (handles conda envs where subprocess has a different PATH)
    benchmark_app = shutil.which("benchmark_app")
    if benchmark_app is None:
        print("[!] benchmark_app not found on PATH. Activate your OpenVINO env first.")
        return
    print(f"  benchmark_app: {benchmark_app}\n")

    models = find_gpu_models(OV_MODELS_DIR, MODEL_PREFIXES)
    if not models:
        print(f"[!] No GPU models found in {OV_MODELS_DIR}/")
        print("    Run convert.py, quantize_uniform.py, and quantize_mixed_gpu.py first.")
        return

    print(f"Found {len(models)} models to benchmark on {DEVICE}:")
    for fname, _ in models:
        print(f"  {fname}")
    print()

    # ── Run benchmarks ───────────────────────────────────────────────────
    for fname, model_path in models:
        blob_name = fname[:-4]
        log_path  = os.path.join(LOG_DIR, f"{blob_name}_{DEVICE}_{HINT}.txt")

        # Skip if already benchmarked successfully (resumable)
        lat, fps = parse_log(log_path)
        if lat and fps:
            print(f"  [SKIP] {blob_name} — already done ({lat:.2f} ms, {fps:.2f} FPS)")
            continue

        cmd = (
            f"{benchmark_app} "
            f"-m {model_path} "
            f"-d {DEVICE} "
            f"-hint {HINT} "
            f"-t {DURATION} "
            f"--inference_only TRUE"
        )

        print(f"{'='*60}")
        print(f"  {blob_name}  →  {DEVICE}")
        print(f"  cmd: {cmd}")
        print(f"{'='*60}")

        with open(log_path, "w") as log_file:
            subprocess.run(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT)

        lat, fps = parse_log(log_path)
        if lat and fps:
            print(f"  → Latency: {lat:.2f} ms  |  Throughput: {fps:.2f} FPS")
        else:
            print(f"  → [!] Could not parse results from {log_path}")
        print()

    # ── Summary CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, "latency_throughput_gpu_report.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "device", "latency_ms", "throughput_fps"])

        for fname, _ in models:
            blob_name = fname[:-4]
            log_path  = os.path.join(LOG_DIR, f"{blob_name}_{DEVICE}_{HINT}.txt")
            lat, fps  = parse_log(log_path)
            writer.writerow([
                blob_name, DEVICE,
                f"{lat:.2f}" if lat else "N/A",
                f"{fps:.2f}" if fps else "N/A",
            ])

    print(f"{'='*60}")
    print(f"  GPU Summary saved: {csv_path}")
    print(f"  Total models benchmarked: {len(models)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
