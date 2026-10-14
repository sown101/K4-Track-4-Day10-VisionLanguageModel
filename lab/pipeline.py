"""Chạy trọn một thí nghiệm: train -> suy luận test -> chấm (kèm --meta). Nhiều thí nghiệm chạy song song, mỗi cái 1 GPU.

    python -m lab.pipeline e1 e2          # Kaggle T4 x2: E1 trên GPU 0, E2 trên GPU 1
    python -m lab.pipeline e3
    python -m lab.pipeline e4_rank_4 e4_rank_64

Bỏ qua thí nghiệm đã có report/evidence/<tên>.json; bỏ qua bước train nếu đã có work/<tên>/run.json.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# tên -> (config, tham số thêm cho lab.train, tham số thêm cho score/infer)
EXPERIMENTS = {
    "e1": ("configs/e1.yaml", [], []),
    "e2": ("configs/e2.yaml", [], []),
    "e3": ("configs/e3.yaml", [], []),
    # E4: đổi đúng MỘT biến so với setup tốt nhất. Sau Version #1, E2 (F1 0.327) > E1 (0.273) nên gốc là E2.
    # data_size: giá trị 1.0 chính là E2 (đã chạy), chỉ cần chạy thêm 0.1.
    "e4_data_0.1": ("configs/e2.yaml", ["--train-fraction", "0.1"], []),
    "e4_rank_4": ("configs/e2.yaml", ["--r", "4"], []),
    "e4_rank_64": ("configs/e2.yaml", ["--r", "64"], []),
    "e4_coord_pixel": ("configs/e2.yaml", ["--coord", "pixel"], ["--coord", "pixel"]),
}


def run_one(name):
    cfg, train_args, coord_args = EXPERIMENTS[name]
    work = Path("work") / name
    evidence = Path("report/evidence") / f"{name}.json"
    py = sys.executable
    if evidence.exists():
        print(f"[{name}] đã có {evidence}, bỏ qua", flush=True)
        return
    if not (work / "run.json").exists():
        subprocess.run([py, "-m", "lab.train", "--config", cfg, "--out", str(work), *train_args], check=True)
    run = json.loads((work / "run.json").read_text(encoding="utf-8"))
    pred = Path("work") / f"{name}.jsonl"
    cmd = [py, "-m", "lab.infer", "--split", "test", "--out", str(pred), "--adapter", str(work / "adapter"),
           "--max-new-tokens", "256", *coord_args]
    if (work / "extra.pt").exists():
        cmd += ["--extra", str(work / "extra.pt")]
    subprocess.run(cmd, check=True)
    subprocess.run([py, "scripts/score.py", "--pred", str(pred), "--name", name.upper(), "--out", str(evidence),
                    *coord_args,
                    "--meta", f"trainable_params={run['trainable_params']}",
                    "--meta", f"peak_vram_gb={run['peak_vram_gb']}",
                    "--meta", f"train_minutes={run['train_minutes']}"], check=True)


def run_parallel(names):
    import torch
    ngpu = max(torch.cuda.device_count(), 1)
    queue, running = list(names), {}
    Path("work").mkdir(exist_ok=True)
    while queue or running:
        free = [g for g in range(ngpu) if g not in running]
        while queue and free:
            g, name = free.pop(0), queue.pop(0)
            log = Path("work") / f"{name}.log"
            fh = open(log, "a", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-m", "lab.pipeline", "--one", name], stdout=fh,
                                 stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
            running[g] = (name, p, fh, log)
            print(f"[GPU {g}] bắt đầu {name} (log: {log})", flush=True)
        time.sleep(60)
        for g, (name, p, fh, log) in list(running.items()):
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            useful = [l for l in lines if not l.startswith("[transformers] Kwargs")]
            print(f"[GPU {g}] {name}: {useful[-1] if useful else '...'}", flush=True)
            if p.poll() is not None:
                fh.close()
                del running[g]
                status = "xong" if p.returncode == 0 else f"LỖI (mã {p.returncode}), xem {log}"
                print(f"[GPU {g}] {name}: {status}", flush=True)
                if p.returncode:
                    print("\n".join(useful[-25:]), flush=True)


def main():
    args = sys.argv[1:]
    if args[:1] == ["--one"]:
        run_one(args[1])
    else:
        unknown = [a for a in args if a not in EXPERIMENTS]
        if unknown:
            raise SystemExit(f"không biết thí nghiệm {unknown}; có: {list(EXPERIMENTS)}")
        run_parallel(args)


if __name__ == "__main__":
    main()
