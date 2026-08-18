#!/usr/bin/env python3
"""Incremental benchmark for the models in this ai_models repo.

Auto-discovers every ncnn model dir (`*/model.ncnn.param`) and every `*.tflite`
under the repo, benchmarks only the (model, threads) pairs **not already in**
`benchmark_results.csv`, appends the new results, and regenerates the charts
**only when new rows were added** (nothing new -> charts untouched).

Each measurement runs in its **own process** so peak RSS (`ru_maxrss`) is clean
per model. FPS = N / wall; CPU% = process_cpu / wall * 100 (100% = one core);
RAM = peak process RSS (includes the runtime).

Usage (run from anywhere; paths are relative to this file):
    python3 bench.py                 # sweep threads 4/8/16, add missing models
    python3 bench.py --threads 4     # single thread count
    python3 bench.py --list          # just show discovered models and CSV status

Needs: ultralytics (+ncnn) for ncnn models, tflite_runtime|tensorflow for tflite,
matplotlib for charts. Easiest in the ros-container.
"""
import argparse
import csv
import glob
import os
import re
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(HERE)               # ai_models repo root
CSV = os.path.join(HERE, "benchmark_results.csv")
FIELDS = ["name", "kind", "threads", "input_px", "dtype",
          "fps", "cpu_pct_across_cores", "peak_ram_mb"]
N, WARMUP = 50, 8


# ----------------------------- worker (one model / process) ------------------
def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _timed(run):
    for _ in range(WARMUP):
        run()
    t0, c0 = time.perf_counter(), time.process_time()
    for _ in range(N):
        run()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    return N / wall, cpu / wall * 100.0


def worker(relpath, kind, threads, imgsz):
    import numpy as np
    path = os.path.join(AI_ROOT, relpath)
    if kind == "ncnn":
        os.environ["OMP_NUM_THREADS"] = str(threads)
        from ultralytics import YOLO
        m = YOLO(path, task="pose")
        img = np.random.randint(0, 255, (imgsz, imgsz, 3), np.uint8)
        run = lambda: m(img, imgsz=imgsz, verbose=False)  # noqa: E731
        input_px, dtype = imgsz, "ncnn"
    else:  # tflite
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        it = Interpreter(model_path=path, num_threads=threads)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        outs = it.get_output_details()
        H, W = int(inp["shape"][1]), int(inp["shape"][2])
        dt = inp["dtype"]
        x = (np.random.rand(1, H, W, 3).astype(dt) if dt == np.float32
             else np.random.randint(0, 255, (1, H, W, 3), dt))

        def run():
            it.set_tensor(inp["index"], x)
            it.invoke()
            for o in outs:
                it.get_tensor(o["index"])
        input_px, dtype = W, str(np.dtype(dt))
    fps, cpu = _timed(run)
    print(f"RESULT input_px={input_px} dtype={dtype} "
          f"fps={fps:.1f} cpu={cpu:.0f} ram={_peak_rss_mb():.0f}")


# ----------------------------- discovery -------------------------------------
def _skip(p):
    rel = os.path.relpath(p, AI_ROOT)
    return (rel.startswith(".git" + os.sep) or rel == ".git"
            or rel.startswith("analytics" + os.sep) or rel == "analytics")


def _ncnn_imgsz(d):
    try:
        import yaml
        meta = yaml.safe_load(open(os.path.join(d, "metadata.yaml")))
        s = meta.get("imgsz")
        return int(s[0] if isinstance(s, (list, tuple)) else s)
    except Exception:
        nums = re.findall(r"\d+", os.path.basename(d))  # last number, e.g. ..._1280_ncnn_model
        return int(nums[-1]) if nums else 640


def clean_name(relpath):
    n = relpath
    for suf in ("_ncnn_model", ".tflite"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    if n.startswith("base_pouso/"):
        n = n[len("base_pouso/"):]
    return n


def discover():
    models = []  # (relpath, kind, imgsz)
    for p in sorted(glob.glob(os.path.join(AI_ROOT, "**", "model.ncnn.param"), recursive=True)):
        d = os.path.dirname(p)
        if _skip(d):
            continue
        models.append((os.path.relpath(d, AI_ROOT), "ncnn", _ncnn_imgsz(d)))
    for p in sorted(glob.glob(os.path.join(AI_ROOT, "**", "*.tflite"), recursive=True)):
        if _skip(p):
            continue
        models.append((os.path.relpath(p, AI_ROOT), "tflite", 0))
    return models


# ----------------------------- orchestrator ----------------------------------
def load_csv():
    rows, done = [], set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add((r["name"], int(r["threads"])))
    return rows, done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--list", action="store_true", help="show models + status, do nothing")
    ap.add_argument("--worker", nargs=4, metavar=("REL", "KIND", "THREADS", "IMGSZ"),
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        rel, kind, th, im = args.worker
        worker(rel, kind, int(th), int(im))
        return

    rows, done = load_csv()
    models = discover()
    print(f"discovered {len(models)} models | {len(done)} (model,threads) already in CSV\n")
    if args.list:
        for rel, kind, imgsz in models:
            name = clean_name(rel)
            have = [t for t in args.threads if (name, t) in done]
            print(f"  {name:38s} {kind:6s} imgsz={imgsz or '-':>4} "
                  f"done@{have or '[]'}")
        return

    new = 0
    for rel, kind, imgsz in models:
        name = clean_name(rel)
        for th in args.threads:
            if (name, th) in done:
                print(f"  skip  {name} @{th}t  (already benchmarked)")
                continue
            print(f"  bench {name} @{th}t ...", flush=True)
            try:
                out = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker",
                     rel, kind, str(th), str(imgsz)],
                    capture_output=True, text=True, timeout=900)
            except subprocess.TimeoutExpired:
                print("        TIMEOUT"); continue
            res = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")]
            if not res:
                tail = (out.stderr.strip().splitlines() or ["(no output)"])[-1]
                print("        FAILED:", tail); continue
            kv = dict(p.split("=", 1) for p in res[0].split()[1:])
            rows.append({"name": name, "kind": kind, "threads": th,
                         "input_px": kv["input_px"], "dtype": kv["dtype"],
                         "fps": kv["fps"], "cpu_pct_across_cores": kv["cpu"],
                         "peak_ram_mb": kv["ram"]})
            done.add((name, th)); new += 1
            print(f"        -> fps={kv['fps']} cpu={kv['cpu']}% ram={kv['ram']}MB "
                  f"input={kv['input_px']} dtype={kv['dtype']}")

    if not new:
        print("\nno new models to benchmark; CSV and charts left untouched.")
        return

    rows.sort(key=lambda r: (r["name"], int(r["threads"])))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    print(f"\nappended {new} new row(s) -> {CSV}")
    print("regenerating charts (new benchmarks present) ...")
    subprocess.run([sys.executable, os.path.join(HERE, "make_charts.py")])


if __name__ == "__main__":
    main()
