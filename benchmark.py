"""
benchmark.py  —  CPU Pipeline

Benchmarks all quantized models on CPU using OpenVINO benchmark_app.
Scans ov_models/ for registered model files (baseline + mixed + uniform),
excluding GPU-specific models (_gpu_point*).

Output:
    log/benchmark_log/{model_name}_CPU_latency.txt   — raw benchmark_app output
    log/benchmark_log/latency_throughput_report.csv   — summary CSV

Usage:
    python benchmark.py
"""

import os
import re
import csv
import shutil
import subprocess

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PREFIXES = [
    "mamba-130m-hf",
    "mamba-1.4b-hf",
]

OV_MODELS_DIR  = "ov_models"
LOG_DIR        = "log/benchmark_log"
DEVICE         = "CPU"
HINT           = "latency"
DURATION       = 30   # seconds per benchmark run

# Regex for parsing benchmark_app output
LATENCY_RE    = re.compile(r"\[ INFO \]\s+Average:\s+([\d\.]+)\s+ms")
THROUGHPUT_RE = re.compile(r"\[ INFO \]\s+Throughput:\s+([\d\.]+)\s+FPS")

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_cpu_models(models_dir, prefixes):
    """
    Find all .xml files in models_dir that belong to registered models.
    Excludes GPU-specific files (*_gpu_point*).
    Returns sorted list of (xml_filename, model_path).
    """
    results = []
    for f in sorted(os.listdir(models_dir)):
        if not f.endswith(".xml"):
            continue
        # Must start with a registered model prefix
        if not any(f.startswith(p) for p in prefixes):
            continue
        # Exclude GPU-specific models
        if "_gpu_point" in f:
            continue
        results.append((f, os.path.join(models_dir, f)))
    return results


def parse_log(log_path):
    """Extract average latency (ms) and throughput (FPS) from benchmark log."""
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

    models = find_cpu_models(OV_MODELS_DIR, MODEL_PREFIXES)
    if not models:
        print(f"[!] No models found in {OV_MODELS_DIR}/ matching prefixes: {MODEL_PREFIXES}")
        print("    Run convert.py, quantize_uniform.py, and quantize_mixed.py first.")
        return

    print(f"Found {len(models)} models to benchmark on {DEVICE}:")
    for fname, _ in models:
        print(f"  {fname}")
    print()

    # ── Run benchmarks ───────────────────────────────────────────────────
    for fname, model_path in models:
        blob_name = fname[:-4]   # strip .xml
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

        # Quick peek at results
        lat, fps = parse_log(log_path)
        if lat and fps:
            print(f"  → Latency: {lat:.2f} ms  |  Throughput: {fps:.2f} FPS")
        else:
            print(f"  → [!] Could not parse results from {log_path}")
        print()

    # ── Extract summary CSV ──────────────────────────────────────────────
    csv_path = os.path.join(LOG_DIR, "latency_throughput_report.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "device", "latency_ms", "throughput_fps"])

        for fname, _ in models:
            blob_name = fname[:-4]
            log_path  = os.path.join(LOG_DIR, f"{blob_name}_{DEVICE}_{HINT}.txt")
            lat, fps  = parse_log(log_path)
            writer.writerow([
                blob_name,
                DEVICE,
                f"{lat:.2f}" if lat else "N/A",
                f"{fps:.2f}" if fps else "N/A",
            ])

    print(f"{'='*60}")
    print(f"  Summary saved: {csv_path}")
    print(f"  Total models benchmarked: {len(models)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
