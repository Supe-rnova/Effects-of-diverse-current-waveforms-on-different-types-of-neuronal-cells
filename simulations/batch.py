"""
Step 7 - batch runner.

    python simulations/batch.py --dry-run          # list what would run + time estimate
    python simulations/batch.py                    # run everything (resumable)
    python simulations/batch.py --models HH STN    # subset of models
    python simulations/batch.py --protocols step sine --limit 5   # quick tests

What it does
  1. measures single-pulse thresholds if missing (simulations/pulse_threshold.py)
  2. builds the task list from config/experiments.py
  3. skips tasks whose result file already exists  ->  safe to stop and restart
  4. splits the rest into chunks (one model/condition/block per chunk, ~5 min each)
     and runs them in parallel worker processes (one model per process)
  5. writes results/data/runs/<task id>.npz, results/data/tasks.json,
     results/data/runs_index.csv and results/logs/step7_batch.log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR, LOGS_DIR, MODEL_NAMES  # noqa: E402

RUNS_DIR = DATA_DIR / "runs"
CHUNK_DIR = RUNS_DIR / "_chunks"
ERR_DIR = LOGS_DIR / "step7_errors"
LOG = LOGS_DIR / "step7_batch.log"
TASKS_FILE = DATA_DIR / "tasks.json"
INDEX_FILE = DATA_DIR / "runs_index.csv"

# wall seconds per simulated second on the user's PC under WSL2 (from Steps 2 and 6);
# only used for chunk sizing and the time estimate
SPEED = {"HH": 0.3, "STN": 3.5, "Purkinje": 22.0, "CA1": 25.0}
CHUNK_WALL_S = 300.0


def result_path(task_id: str) -> Path:
    return RUNS_DIR / f"{task_id}.npz"


def log(msg: str):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ===================================================================== worker
def build_waveform(components):
    from waveforms import generator as g
    w = None
    for kind, params in components:
        part = g.make(kind, **params)
        w = part if w is None else w + part
    if len(components) > 1:
        w.kind = components[0][0] + "+bg"
        w.params = {"components": [[k, p] for k, p in components]}
    return w


def worker(chunk_file: str):
    from analysis.spikes import detect_spikes
    from simulations.injection import Simulator, save_result
    tasks = json.loads(Path(chunk_file).read_text())
    sim = Simulator(tasks[0]["model"])
    for t in tasks:
        out = result_path(t["id"])
        if out.exists():
            print(f"SKIP {t['id']}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            sim.prepare(t["condition"], t["block"])
            w = build_waveform(t["components"])
            extra = {k: t[k] for k in ("id", "protocol", "group", "level", "trial")}
            res = sim.run(w, site=t["site"], lead_in=t["lead_in"], extra_meta=extra)
            s = detect_spikes(res["t"], res["v"]["soma"])
            res["meta"]["spike_times_soma"] = s.times.tolist()
            res["meta"]["spike_thresholds_soma"] = s.thresholds.tolist()
            tmp = out.with_name(out.name + ".tmp.npz")
            save_result(tmp, res)
            os.replace(tmp, out)                     # atomic: no half-written results
            print(f"DONE {time.perf_counter() - t0:.1f} {t['id']}", flush=True)
        except Exception:
            ERR_DIR.mkdir(parents=True, exist_ok=True)
            (ERR_DIR / (t["id"].replace("/", "__") + ".log")).write_text(traceback.format_exc())
            print(f"ERROR {t['id']}", flush=True)


# ===================================================================== parent
def make_chunks(tasks):
    groups = defaultdict(list)
    for t in tasks:
        groups[(t["model"], t["condition"], t["block"])].append(t)
    chunks = []
    for (model, _, _), ts in groups.items():
        cur, cost = [], 0.0
        for t in ts:
            c = t["sim_ms"] / 1000.0 * SPEED.get(model, 10.0)
            if cur and cost + c > CHUNK_WALL_S:
                chunks.append((cost, cur))
                cur, cost = [], 0.0
            cur.append(t)
            cost += c
        if cur:
            chunks.append((cost, cur))
    chunks.sort(key=lambda x: -x[0])                 # longest first = better load balance
    return chunks


def _reader(proc, q):
    for line in proc.stdout:
        q.put(line.rstrip())
    proc.stdout.close()


def run_chunks(chunks, n_total, max_workers):
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for k, (_, ts) in enumerate(chunks):
        f = CHUNK_DIR / f"chunk_{k:04d}.json"
        f.write_text(json.dumps(ts))
        files.append(f)
    q = queue.Queue()
    pending, running = list(files), []
    done = errors = 0
    t_start = last_report = time.perf_counter()
    while pending or running:
        while pending and len(running) < max_workers:
            f = pending.pop(0)
            p = subprocess.Popen([sys.executable, __file__, "--worker", str(f)], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            threading.Thread(target=_reader, args=(p, q), daemon=True).start()
            running.append((p, f))
        try:
            while True:
                line = q.get_nowait()
                if line.startswith("DONE") or line.startswith("SKIP"):
                    done += 1
                elif line.startswith("ERROR"):
                    errors += 1
                    log(f"  error in {line[6:]} (traceback in {ERR_DIR})")
        except queue.Empty:
            pass
        for p, f in list(running):
            if p.poll() is not None:
                err = p.stderr.read()
                if p.returncode != 0:
                    log(f"  worker for {f.name} exited with code {p.returncode}:\n{err[-2000:]}")
                running.remove((p, f))
        now = time.perf_counter()
        if now - last_report > 30 or (not pending and not running):
            el = now - t_start
            eta = el / done * (n_total - done) if done else float("nan")
            log(f"  progress {done}/{n_total} runs ({100 * done / max(n_total, 1):.0f} %), "
                f"{errors} errors, elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min, "
                f"{len(running)} workers busy")
            last_report = now
        time.sleep(0.5)
    return done, errors


def write_index(tasks):
    TASKS_FILE.write_text(json.dumps(tasks))
    with open(INDEX_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "model", "condition", "block", "site", "protocol", "group", "level",
                    "trial", "lead_in", "file", "done"])
        for t in tasks:
            p = result_path(t["id"])
            w.writerow([t["id"], t["model"], t["condition"], t["block"] or "", t["site"], t["protocol"],
                        t["group"], json.dumps(t["level"]), t["trial"] if t["trial"] is not None else "",
                        t["lead_in"], str(p.relative_to(DATA_DIR)), int(p.exists())])


def main():
    from config.experiments import build_tasks
    from simulations import pulse_threshold
    from simulations.injection import load_characterisation

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max tasks per model (testing)")
    ap.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    models = a.models or MODEL_NAMES

    char = load_characterisation()
    pthr_file = pulse_threshold.OUT
    pthr = json.loads(pthr_file.read_text()) if pthr_file.exists() else {}
    missing = [m for m in models if m not in pthr]
    if missing and not a.dry_run:
        log(f"measuring single-pulse thresholds for {missing} ...")
        pthr = pulse_threshold.main(missing)

    all_tasks = build_tasks(char, pthr, models)
    tasks = [t for t in all_tasks if not a.protocols or t["protocol"] in a.protocols]
    if a.limit:
        per = defaultdict(int)
        sel = []
        for t in tasks:
            if per[t["model"]] < a.limit:
                sel.append(t)
                per[t["model"]] += 1
        tasks = sel
    todo = [t for t in tasks if not result_path(t["id"]).exists()]

    # summary + estimate
    cpu = defaultdict(float)
    n = defaultdict(int)
    for t in todo:
        cpu[t["model"]] += t["sim_ms"] / 1000.0 * SPEED.get(t["model"], 10.0)
        n[t["model"]] += 1
    total_cpu = sum(cpu.values())
    log(f"{len(tasks)} tasks selected, {len(tasks) - len(todo)} already done, {len(todo)} to run")
    for m in models:
        if n[m]:
            log(f"  {m:9s} {n[m]:4d} runs  ~{cpu[m] / 3600:.2f} CPU-h")
    est = max(total_cpu / a.max_workers, max(cpu.values(), default=0) / max(1, a.max_workers))
    log(f"  estimated wall time with {a.max_workers} workers: ~{est / 60:.0f} min "
        f"(total {total_cpu / 3600:.1f} CPU-h)")
    if missing and a.dry_run:
        log(f"  note: pulse thresholds not measured yet for {missing} - their pulse runs are not counted")
    if a.dry_run:
        by = defaultdict(int)
        for t in todo:
            by[(t["model"], t["condition"], t["block"] or "ctrl", t["site"], t["protocol"])] += 1
        for k in sorted(by):
            print("   ", "/".join(k), by[k])
        return

    if todo:
        t0 = time.perf_counter()
        done, errors = run_chunks(make_chunks(todo), len(todo), a.max_workers)
        log(f"finished {done} runs with {errors} errors in {(time.perf_counter() - t0) / 60:.1f} min")
    write_index(all_tasks)
    n_done = sum(result_path(t["id"]).exists() for t in all_tasks)
    log(f"index: {INDEX_FILE}  ({n_done}/{len(all_tasks)} runs complete)")
    log("BATCH COMPLETE" if n_done == len(all_tasks) else
        "BATCH INCOMPLETE - rerun the same command to finish the remaining runs")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
