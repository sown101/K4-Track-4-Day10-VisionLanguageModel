"""Bước 4–6: dữ liệu huấn luyện (chat template + mask loss), setup LoRA/đóng băng, vòng huấn luyện QLoRA.

    python -m lab.train --config configs/e1.yaml --out work/e1
    python -m lab.train --config configs/e1.yaml --out work/e4_rank_4 --r 4          # E4: đổi đúng một biến
    python -m lab.train --config configs/e1.yaml --out work/e1_dbg --max-steps 5 --check   # gỡ lỗi nhanh

Đầu ra trong --out: adapter/ (LoRA, peft), extra.pt (projector huấn luyện đầy đủ, E2/E3), run.json.
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import yaml

# Tên module lấy từ cây Qwen3VLForConditionalGeneration (transformers 5.x); peft so khớp bằng re.fullmatch
LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
EXPECTED_MODULES = {"llm": 252, "llm+vit": 348}   # 36×7 và 36×7 + 24×4


# ---------------------------------------------------------------- Bước 4: encode + collate
def encode_example(processor, image, prompt, answer):
    """Trả dict tensor (batch=1) gồm labels: chỉ phần trả lời (+ <|im_end|>) được tính loss."""
    import torch
    user = {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    ids = full_enc["input_ids"]
    if not torch.equal(ids[0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ:\n"
                         + repr(processor.tokenizer.decode(ids[0, :n + 10])))
    labels = ids.clone()
    labels[:, :n] = -100                                  # prompt + 220 token ảnh: không học
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (ids[0] == im_end).nonzero().flatten()
    labels[:, ends[-1] + 1:] = -100                       # "\n" sau <|im_end|> cuối: không học
    out = {k: v for k, v in full_enc.items()}
    out["labels"] = labels
    return out


def collate(batch, pad_id):
    """Pad các tensor theo chuỗi (right-pad); pixel_values/image_grid_thw nối theo chiều 0, không pad."""
    import torch
    L = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for k in batch[0]:
        if k in ("pixel_values", "image_grid_thw"):
            out[k] = torch.cat([b[k] for b in batch], dim=0)
            continue
        fill = {"input_ids": pad_id, "labels": -100}.get(k, 0)   # attention_mask, mm_token_type_ids -> 0
        rows = []
        for b in batch:
            t = b[k]
            pad = torch.full((1, L - t.shape[1]), fill, dtype=t.dtype)
            rows.append(torch.cat([t, pad], dim=1))
        out[k] = torch.cat(rows, dim=0)
    return out


class TrainSet:
    def __init__(self, rows, data_dir, processor, prompt, coord):
        self.rows, self.data, self.processor, self.prompt, self.coord = rows, Path(data_dir), processor, prompt, coord

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        from lab.labels import objects_to_answer
        r = self.rows[i]
        img = Image.open(self.data / r["image_path"]).convert("RGB")
        ans = objects_to_answer(r["objects"], self.coord, r["width"], r["height"])
        return encode_example(self.processor, img, self.prompt, ans)


def check_example(processor, ex):
    """Kiểm tra bước 4: số token ảnh và phần được tính loss."""
    image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_img = int((ex["input_ids"] == image_pad).sum())
    learn = ex["input_ids"][ex["labels"] != -100]
    print(f"token ảnh = {n_img} (640x360 phải là 220) | tổng token = {ex['input_ids'].shape[1]} "
          f"| token tính loss = {len(learn)}", flush=True)
    print("phần tính loss:", repr(processor.tokenizer.decode(learn)), flush=True)
    return n_img


# ---------------------------------------------------------------- Bước 5: LoRA / đóng băng
def apply_setup(model, lora_cfg, train_projector):
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()

    targets = lora_cfg["targets"]
    regex = LLM_TARGETS if targets == "llm" else f"({LLM_TARGETS}|{VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    print(f"LoRA targets={targets}: {len(matched)} module (kỳ vọng {EXPECTED_MODULES[targets]})", flush=True)
    if len(matched) != EXPECTED_MODULES[targets]:
        raise ValueError(f"số module khớp sai: {len(matched)}, ví dụ {matched[:3]}")

    model = get_peft_model(model, LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=regex, task_type="CAUSAL_LM"))

    extra = []
    if train_projector:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()   # master weight fp32: GradScaler không unscale được grad fp16
                p.requires_grad = True
                extra.append(n)
    lora_n = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    proj_n = sum(p.numel() for n, p in model.named_parameters() if n in set(extra))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"tham số huấn luyện: {total:,} (LoRA {lora_n:,} + projector {proj_n:,})", flush=True)
    return model, extra, total


def param_counts(model_id="Qwen/Qwen3-VL-4B-Instruct"):
    """Số tham số từng khối (Câu 1), dựng model trên thiết bị 'meta' nên không tốn RAM.
    Không đếm trên model 4-bit: Params4bit đóng gói 2 giá trị/byte nên numel() chỉ còn một nửa."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(model_id))

    def n(prefix):
        return sum(p.numel() for k, p in model.named_parameters() if k.startswith(prefix))
    merger, deep = n("model.visual.merger."), n("model.visual.deepstack_merger_list.")
    vit = n("model.visual.") - merger - deep
    return {"ViT (blocks + patch_embed + pos_embed)": vit, "visual.merger": merger,
            "visual.deepstack_merger_list": deep, "LLM (language_model, gồm embedding)": n("model.language_model."),
            "lm_head": n("lm_head."), "tổng": sum(p.numel() for p in model.parameters()),
            "tied_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False))}


# ---------------------------------------------------------------- Bước 6: vòng huấn luyện
def save_ckpt(model, extra, out):
    import torch
    model.save_pretrained(out / "adapter")
    if extra:
        params = dict(model.named_parameters())
        torch.save({n: params[n].detach().cpu() for n in extra}, out / "extra.pt")


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    from lab.infer import load_model
    from lab.labels import build_prompt
    from scripts.labcommon import read_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lora = dict(cfg["lora"])
    if args.r:                                   # E4 rank: alpha = 2r
        lora["r"], lora["alpha"] = args.r, 2 * args.r
    max_steps = args.max_steps or cfg["max_steps"]
    bs, accum, lr, seed = cfg["batch_size"], cfg["grad_accum"], cfg["lr"], cfg["seed"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    rows = read_jsonl(Path(args.data) / "splits" / "train.jsonl")
    if args.train_fraction < 1:
        rows = random.Random(seed).sample(rows, max(1, round(len(rows) * args.train_fraction)))
    print(f"{cfg['name']}: {len(rows)} ảnh train, coord={args.coord}, lora={lora}, "
          f"projector={cfg['train_projector']}, steps={max_steps}", flush=True)

    model, processor = load_model()
    model, extra, n_train = apply_setup(model, lora, cfg["train_projector"])
    prompt = build_prompt(args.coord, rows[0]["width"], rows[0]["height"])
    ds = TrainSet(rows, args.data, processor, prompt, args.coord)
    n_img = check_example(processor, ds[0])
    if args.check and n_img != 220:
        raise ValueError(f"token ảnh = {n_img}, kỳ vọng 220")

    pad_id = processor.tokenizer.pad_token_id
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True,
                    collate_fn=lambda b: collate(b, pad_id),
                    generator=torch.Generator().manual_seed(seed))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.05 * max_steps), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()

    log, step, t0, it = [], 0, time.time(), iter(dl)
    while step < max_steps:
        loss_sum = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        log.append({"step": step, "loss": loss_sum, "grad_norm": gnorm, "lr": sched.get_last_lr()[0]})
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step}/{max_steps} loss={loss_sum:.4f} gnorm={gnorm:.2f} "
                  f"lr={sched.get_last_lr()[0]:.2e} | {el / 60:.1f} phút | còn ~{el / step * (max_steps - step) / 60:.1f} phút",
                  flush=True)
        if step % 100 == 0 or step == max_steps:
            save_ckpt(model, extra, out)

    run = {"name": cfg["name"], "config": args.config, "lora": lora, "train_projector": cfg["train_projector"],
           "coord": args.coord, "train_images": len(rows), "max_steps": max_steps,
           "trainable_params": n_train,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "log": log}
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print(f"xong: trainable={n_train:,} peak_vram={run['peak_vram_gb']} GB "
          f"train_minutes={run['train_minutes']}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--r", type=int, default=0, help="E4: ghi đè LoRA rank (alpha = 2r)")
    ap.add_argument("--train-fraction", type=float, default=1.0, help="E4: tỉ lệ ảnh train")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="dừng nếu số token ảnh khác 220")
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
