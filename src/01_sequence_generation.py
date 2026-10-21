#!/usr/bin/env python
"""Targeted short-peptide generation with ProtGPT2.

Input
-----
A FASTA or CSV file of seed peptides (known antimicrobial peptides). For CSV,
the column named ``sequence`` is used.

Output
------
``<out-dir>/generated_sequences.fasta``
    Unique peptides in FASTA format. Headers carry ``len``, ``seed`` and ``batch``.
``<out-dir>/generation_progress.json``
    Checkpoint written every ``checkpoint_every_batches`` batches.
``<out-dir>/generation_stats.json``
    Run parameters and a length distribution.

Example
-------
python src/01_sequence_generation.py \
    --seeds data/known_amps_example.csv \
    --out-dir runs/01_generation \
    --target 200 \
    --device cpu

The generation strategy uses an N-terminal seed prefix as the prompt, samples
several continuations per prompt, enforces the configured length range and
removes duplicates against both the seed set and previously generated peptides.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import (
    clean_sequence,
    ensure_dir,
    get_logger,
    load_config,
    peptide_id,
    read_fasta,
)
from physchem import STD_AA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", required=True, help="FASTA or CSV of seed peptides")
    parser.add_argument("--out-dir", default="runs/01_generation")
    parser.add_argument("--model", default=None, help="HF model id or local directory")
    parser.add_argument("--target", type=int, default=None, help="unique sequences to generate")
    parser.add_argument("--max-batches", type=int, default=0, help="0 = no limit")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_seeds(path: str) -> list[str]:
    """Read seed sequences from FASTA or CSV."""
    if path.lower().endswith((".fa", ".fasta", ".faa")):
        sequences = [sequence for _, sequence in read_fasta(path)]
    else:
        sequences = []
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sequence" not in reader.fieldnames:
                raise ValueError("CSV seed file must contain a 'sequence' column")
            for row in reader:
                sequence = clean_sequence(row["sequence"])
                if sequence:
                    sequences.append(sequence)
    unique = []
    seen = set()
    for sequence in sequences:
        if sequence not in seen:
            seen.add(sequence)
            unique.append(sequence)
    return unique


def load_existing(fasta_path: Path) -> tuple[set[str], int]:
    """Recover generated sequences and the largest numeric id from a checkpoint."""
    if not fasta_path.exists():
        return set(), 0
    seen: set[str] = set()
    max_id = 0
    for header, sequence in read_fasta(str(fasta_path)):
        seen.add(sequence)
        token = peptide_id(header)
        if token.startswith("gen_"):
            try:
                max_id = max(max_id, int(token.split("_", 1)[1]))
            except ValueError:
                pass
    return seen, max_id


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    gen_cfg = cfg.get("generation", {})
    seed_value = args.seed if args.seed is not None else int(cfg.get("project", {}).get("seed", 0))
    min_len = int(gen_cfg.get("min_length", 10))
    max_len = int(gen_cfg.get("max_length", 50))
    prefix_len = int(gen_cfg.get("prefix_length", 3))
    target = int(args.target or gen_cfg.get("target_sequences", 100000))
    model_name = args.model or cfg.get("paths", {}).get("protgpt2_model", "nferruz/ProtGPT2")

    out_dir = ensure_dir(args.out_dir)
    fasta_path = out_dir / "generated_sequences.fasta"
    progress_path = out_dir / "generation_progress.json"
    stats_path = out_dir / "generation_stats.json"
    logger = get_logger("generation", str(out_dir / "generation.log"))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Sequence generation requires 'torch' and 'transformers'. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    seeds = load_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seed sequences were loaded")
    seed_set = set(seeds)
    prefix_pool = sorted({s[:prefix_len] for s in seeds if len(s) >= prefix_len})
    if not prefix_pool:
        raise SystemExit("No seed is long enough to build a prompt prefix")

    seen, max_id = load_existing(fasta_path)
    if len(seen) >= target:
        logger.info("Target already reached: %d unique sequences", len(seen))
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=args.local_files_only, torch_dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.to(device).eval()

    prompts_per_batch = int(gen_cfg.get("prompts_per_batch", 20))
    samples_per_prompt = int(gen_cfg.get("samples_per_prompt", 10))
    token_choices = list(gen_cfg.get("max_new_token_choices", [12, 15, 18, 20]))
    checkpoint_every = int(gen_cfg.get("checkpoint_every_batches", 10))
    current_prompts = prompts_per_batch

    started = time.time()
    batch_index = 0
    next_id = max_id + 1
    written = 0
    attempted_total = 0
    accepted_total = 0
    stop_reason = "target_reached"

    with open(fasta_path, "a", encoding="utf-8") as handle:
        while len(seen) < target:
            if args.max_batches and batch_index >= args.max_batches:
                stop_reason = "max_batches_reached"
                break
            batch_index += 1
            n_prompts = min(current_prompts, len(prefix_pool))
            prompts = random.sample(prefix_pool, n_prompts)
            max_new_tokens = random.choice(token_choices)
            try:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        num_return_sequences=samples_per_prompt,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=float(gen_cfg.get("temperature", 0.85)),
                        top_p=float(gen_cfg.get("top_p", 0.9)),
                        top_k=int(gen_cfg.get("top_k", 40)),
                        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.2)),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("CUDA out of memory; clearing cache and reducing the batch")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    current_prompts = max(1, current_prompts // 2)
                    batch_index -= 1
                    continue
                logger.error("Batch %d failed: %s", batch_index, exc)
                continue

            for index, row in enumerate(output):
                prompt = prompts[min(index // samples_per_prompt, len(prompts) - 1)]
                sequence = clean_sequence(tokenizer.decode(row, skip_special_tokens=True))
                if len(sequence) > max_len:
                    sequence = sequence[:max_len]
                if len(sequence) < min_len or any(a not in STD_AA for a in sequence):
                    continue
                attempted_total += 1
                if sequence in seen or sequence in seed_set:
                    continue
                seen.add(sequence)
                accepted_total += 1
                handle.write(
                    f">gen_{next_id:06d} len={len(sequence)} seed={prompt} batch={batch_index}\n"
                    f"{sequence}\n"
                )
                next_id += 1
                written += 1
            handle.flush()
            del output

            if batch_index % checkpoint_every == 0:
                checkpoint = {
                    "target": target,
                    "unique": len(seen),
                    "written": written,
                    "batches": batch_index,
                    "elapsed_s": round(time.time() - started, 1),
                    "params": gen_cfg,
                }
                progress_path.write_text(
                    json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                rate = len(seen) / max(time.time() - started, 1e-9)
                logger.info(
                    "Batch %d | unique %d/%d | %.1f seq/s",
                    batch_index, len(seen), target, rate,
                )

    lengths = [len(sequence) for sequence in seen]
    distribution = Counter((length // 5) * 5 for length in lengths)
    stats = {
        "target": target,
        "unique_sequences": len(seen),
        "generated_this_run": written,
        "batches": batch_index,
        "elapsed_s": round(time.time() - started, 1),
        "stop_reason": stop_reason,
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": round(float(np.mean(lengths)), 2) if lengths else None,
        "length_bins": {f"{k}-{k + 4}": v for k, v in sorted(distribution.items())},
        "duplicate_rate": round(1 - accepted_total / max(attempted_total, 1), 4),
        "params": gen_cfg,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Finished: %d unique sequences -> %s", len(seen), fasta_path)


if __name__ == "__main__":
    main()
