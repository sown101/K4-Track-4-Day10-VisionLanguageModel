"""Bước 3 / 7: suy luận Qwen3-VL-4B (4-bit NF4) trên một split -> predictions.jsonl.

Mỗi dòng output: {"image_id": ..., "raw": <chuỗi model sinh ra>} — đúng định dạng scripts/score.py.

    python -m lab.infer --split test --out work/e0.jsonl                 # E0 zero-shot
    python -m lab.infer --split test --out work/e0.jsonl --gpus 2        # chia đôi test cho 2 GPU (Kaggle T4 x2)
    python -m lab.infer --split test --out work/e1.jsonl --adapter work/e1/adapter [--extra work/e1/extra.pt]

Ghi nối tiếp và bỏ qua ảnh đã có trong --out, nên chạy lại sau khi mất session sẽ tiếp tục chỗ cũ.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def load_model(adapter=None, extra=None, model_id=MODEL_ID):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # ViT (gồm cả projector merger/deepstack) và lm_head giữ fp16: E2/E3 cần huấn luyện đầy đủ merger
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    if extra:
        # projector huấn luyện đầy đủ (E2/E3) không nằm trong adapter LoRA -> nạp lại riêng
        state = torch.load(extra, map_location="cpu")
        params = dict(model.named_parameters())
        missing = [k for k in state if k not in params]
        if missing:
            raise KeyError(f"{len(missing)} tham số extra không khớp model, ví dụ {missing[:3]}")
        with torch.no_grad():
            for k, v in state.items():
                params[k].data.copy_(v.to(params[k].dtype))
        print(f"nạp lại {len(state)} tensor projector từ {extra}", flush=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"   # sinh tiếp sau prompt: phải pad bên trái
    return model, processor


def build_messages(image, prompt):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]


def generate(model, processor, images, prompt, max_new_tokens):
    import torch
    msgs = [build_messages(im, prompt) for im in images]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)


def _done_ids(path):
    if not Path(path).exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["image_id"])
            except (ValueError, KeyError):
                pass   # dòng ghi dở khi bị ngắt
    return ids


def run(args):
    from PIL import Image
    from lab.labels import build_prompt
    from scripts.labcommon import read_jsonl

    data = Path(args.data)
    rows = read_jsonl(data / "splits" / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.num_shards]
    done = _done_ids(args.out)
    todo = [r for r in rows if r["image_id"] not in done]
    print(f"[shard {args.shard}/{args.num_shards}] {len(rows)} ảnh, đã có {len(rows) - len(todo)}, "
          f"còn {len(todo)}", flush=True)
    if not todo:
        return
    model, processor = load_model(args.adapter, args.extra)
    prompt = build_prompt(args.coord, todo[0]["width"], todo[0]["height"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            images = [Image.open(data / r["image_path"]).convert("RGB") for r in batch]
            raws = generate(model, processor, images, prompt, args.max_new_tokens)
            for r, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": r["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(batch)
            el = time.time() - t0
            print(f"[shard {args.shard}] {n}/{len(todo)} ảnh | {el / 60:.1f} phút | "
                  f"còn ~{el / n * (len(todo) - n) / 60:.1f} phút", flush=True)
    import torch
    print(f"[shard {args.shard}] xong. VRAM đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
          flush=True)


def run_multi_gpu(args):
    """Chạy song song --gpus tiến trình, mỗi tiến trình 1 GPU và 1 phần split, rồi gộp file."""
    out = Path(args.out)
    procs, parts, logs = [], [], []
    for g in range(args.gpus):
        part = out.with_suffix(f".part{g}.jsonl")
        log = out.with_suffix(f".part{g}.log")
        cmd = [sys.executable, "-m", "lab.infer", "--split", args.split, "--out", str(part),
               "--batch", str(args.batch), "--max-new-tokens", str(args.max_new_tokens),
               "--coord", args.coord, "--data", args.data, "--limit", str(args.limit),
               "--shard", str(g), "--num-shards", str(args.gpus)]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        if args.extra:
            cmd += ["--extra", args.extra]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        fh = open(log, "a", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))
        parts.append(part)
        logs.append((log, fh))
    print("log:", ", ".join(str(l) for l, _ in logs), flush=True)
    last = 0
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        if time.time() - last > 60:
            last = time.time()
            for log, _ in logs:
                tail = Path(log).read_text(encoding="utf-8", errors="replace").strip().splitlines()
                if tail:
                    print(tail[-1], flush=True)
    for _, fh in logs:
        fh.close()
    codes = [p.returncode for p in procs]
    if any(codes):
        for log, _ in logs:
            print(f"--- {log} ---\n" + "\n".join(Path(log).read_text(encoding="utf-8").splitlines()[-30:]))
        raise SystemExit(f"tiến trình con lỗi, mã thoát {codes}")
    seen, rows = set(), []
    for part in parts:
        with open(part, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["image_id"] not in seen:
                    seen.add(r["image_id"])
                    rows.append(r)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gộp {len(rows)} dòng -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N ảnh đầu (gỡ lỗi)")
    ap.add_argument("--adapter")
    ap.add_argument("--extra", help="file .pt chứa tham số projector đã huấn luyện (E2/E3)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)
    if args.gpus > 1:
        run_multi_gpu(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
