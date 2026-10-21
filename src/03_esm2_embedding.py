#!/usr/bin/env python
"""ESM-2 sequence embeddings for a peptide library.

Input
-----
A FASTA file of peptides (typically ``screened_physchem.fasta``).

Output
------
``<out-dir>/esm2_embeddings.npy``
    ``float32`` matrix of shape ``(n_sequences, hidden_size)``.
``<out-dir>/esm2_embedding_index.csv``
    Row index, peptide id, length and sequence for each matrix row.
``<out-dir>/embedding_progress.json``
    Checkpoint for resuming an interrupted run.

Embeddings are mean-pooled over residue positions, excluding the BOS and EOS
tokens. The default model is ``facebook/esm2_t33_650M_UR50D`` (1280 dimensions).
The script works on CPU or GPU; GPU is strongly recommended for large libraries.

Example
-------
python src/03_esm2_embedding.py \
    --input runs/02_filter/screened_physchem.fasta \
    --out-dir runs/03_embedding \
    --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from common import ensure_dir, get_logger, load_config, peptide_id, read_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="input FASTA")
    parser.add_argument("--out-dir", default="runs/03_embedding")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="HF model id or local directory")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    emb_cfg = cfg.get("embedding", {})
    model_name = args.model or emb_cfg.get("model") or cfg.get("paths", {}).get(
        "esm2_model", "facebook/esm2_t33_650M_UR50D"
    )
    batch_size = int(args.batch_size or emb_cfg.get("batch_size", 100))
    max_length = int(args.max_length or emb_cfg.get("max_length", 1024))

    out_dir = ensure_dir(args.out_dir)
    logger = get_logger("esm2_embedding", str(out_dir / "embedding.log"))
    records = read_fasta(args.input)
    if not records:
        raise SystemExit("No sequences were found in the input FASTA")
    identifiers = []
    used = set()
    for index, (header, _) in enumerate(records, start=1):
        identifier = peptide_id(header) or f"pep_{index:06d}"
        if identifier in used:
            identifier = f"{identifier}_{index:06d}"
        used.add(identifier)
        identifiers.append(identifier)
    sequences = [sequence for _, sequence in records]

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ESM-2 embedding requires 'torch' and 'transformers'. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    if device == "cpu":
        logger.warning("Running ESM-2 on CPU; use --device cuda for large libraries")

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(
        model_name, local_files_only=args.local_files_only, torch_dtype=torch.float32
    )
    model.to(device).eval()
    hidden_size = int(model.config.hidden_size)
    total = len(sequences)

    npy_path = out_dir / "esm2_embeddings.npy"
    progress_path = out_dir / "embedding_progress.json"
    index_path = out_dir / "esm2_embedding_index.csv"

    done = 0
    if not args.overwrite and npy_path.exists() and progress_path.exists():
        try:
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
            if previous.get("n_sequences") == total and previous.get("hidden_size") == hidden_size:
                done = int(previous.get("completed", 0))
                logger.info("Resuming from %d/%d sequences", done, total)
        except (ValueError, KeyError):
            done = 0

    if done == 0 or args.overwrite:
        matrix = np.lib.format.open_memmap(
            npy_path, mode="w+", dtype=np.float32, shape=(total, hidden_size)
        )
        done = 0
    else:
        matrix = np.lib.format.open_memmap(npy_path, mode="r+")

    started = time.time()
    position = done
    current_batch = batch_size
    while position < total:
        chunk = sequences[position:position + current_batch]
        try:
            encoded = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                output = model(**encoded)
            hidden = output.last_hidden_state.float()
            attention = encoded["attention_mask"]
            mask = attention.unsqueeze(-1).float()
            lengths = attention.sum(dim=1)
            for row_index in range(len(chunk)):
                mask[row_index, 0, :] = 0
                mask[row_index, int(lengths[row_index]) - 1, :] = 0
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            matrix[position:position + len(chunk)] = pooled.cpu().numpy()
            matrix.flush()
            position += len(chunk)
            del output, hidden, pooled
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device == "cuda":
                    torch.cuda.empty_cache()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
                logger.warning("Out of memory; reducing batch size to %d", current_batch)
                continue
            logger.error("Batch starting at %d failed: %s", position, exc)
            for offset, sequence in enumerate(chunk):
                try:
                    single = tokenizer(
                        [sequence], return_tensors="pt", truncation=True,
                        max_length=max_length,
                    ).to(device)
                    with torch.no_grad():
                        single_output = model(**single)
                    length = int(single["attention_mask"][0].sum())
                    matrix[position + offset] = (
                        single_output.last_hidden_state[0, 1:length - 1].float().mean(dim=0)
                        .cpu().numpy()
                    )
                except Exception as inner:  # pragma: no cover - defensive
                    logger.error("Single-sequence fallback failed at %d: %s", position + offset, inner)
                    matrix[position + offset] = np.nan
            matrix.flush()
            position += len(chunk)

        if position % 1000 < current_batch or position == total:
            progress_path.write_text(
                json.dumps({
                    "n_sequences": total,
                    "hidden_size": hidden_size,
                    "completed": position,
                    "elapsed_s": round(time.time() - started, 1),
                }, indent=2),
                encoding="utf-8",
            )
            logger.info("Embedded %d/%d sequences", position, total)

    matrix.flush()
    with open(index_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "peptide_id", "length", "sequence"])
        for row_index, (identifier, sequence) in enumerate(zip(identifiers, sequences)):
            writer.writerow([row_index, identifier, len(sequence), sequence])

    array = np.load(npy_path, mmap_mode="r")
    n_nan = int(np.isnan(array).any(axis=1).sum())
    progress_path.write_text(
        json.dumps({
            "n_sequences": total,
            "hidden_size": hidden_size,
            "completed": total,
            "nan_rows": n_nan,
            "elapsed_s": round(time.time() - started, 1),
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("Finished: shape=%s | NaN rows=%d | %s", array.shape, n_nan, npy_path)


if __name__ == "__main__":
    main()
