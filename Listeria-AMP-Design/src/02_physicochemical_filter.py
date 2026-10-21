#!/usr/bin/env python
"""Rule-based physicochemical filtering of generated peptides.

Input
-----
A FASTA file of candidate peptides (for example, the output of
``01_sequence_generation.py``). An optional FASTA/CSV file of known peptides
can be supplied to remove exact duplicates of the reference set.

Rules
-----
1. length within ``[min_length, max_length]``
2. only the 20 standard amino acids
3. no homopolymer run of ``max_homopolymer_run`` or more residues
4. net charge at pH 7.4 within ``[charge_min, charge_max]``
5. GRAVY within ``[gravy_min, gravy_max]``
6. no exact duplicate within the input or against the exclusion set

Output
------
``<out-dir>/screened_physchem.fasta``
``<out-dir>/screened_physchem.csv``   full descriptor table
``<out-dir>/filter_summary.json``     counts and rejection reasons

Example
-------
python src/02_physicochemical_filter.py \
    --input data/example_input.fasta \
    --exclude data/known_amps_example.csv \
    --out-dir runs/02_filter
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter

from common import clean_sequence, ensure_dir, get_logger, load_config, peptide_id, read_fasta, write_fasta
from physchem import STD_AA, all_descriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="input FASTA")
    parser.add_argument("--out-dir", default="runs/02_filter")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exclude", default=None, help="optional FASTA/CSV of peptides to exclude")
    return parser.parse_args()


def load_exclusion(path: str | None) -> set[str]:
    """Load sequences that should not pass the filter (e.g. known peptides)."""
    if not path:
        return set()
    if path.lower().endswith((".fa", ".fasta", ".faa")):
        return {sequence for _, sequence in read_fasta(path)}
    sequences = set()
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError("CSV exclusion file must contain a 'sequence' column")
        for row in reader:
            sequence = clean_sequence(row["sequence"])
            if sequence:
                sequences.add(sequence)
    return sequences


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    params = cfg.get("physchem", {})
    min_len = int(params.get("min_length", 10))
    max_len = int(params.get("max_length", 50))
    charge_min = float(params.get("charge_min", -2.0))
    charge_max = float(params.get("charge_max", 6.0))
    gravy_min = float(params.get("gravy_min", -1.5))
    gravy_max = float(params.get("gravy_max", 1.0))
    max_run = int(params.get("max_homopolymer_run", 5))

    out_dir = ensure_dir(args.out_dir)
    logger = get_logger("physchem_filter", str(out_dir / "filter.log"))
    records = read_fasta(args.input)
    exclusion = load_exclusion(args.exclude)
    logger.info("Loaded %d candidate peptides (%d excluded reference sequences)", len(records), len(exclusion))

    reasons = Counter()
    seen: set[str] = set()
    passed: list[tuple[str, str]] = []
    used_ids: set[str] = set()

    for index, (header, raw_sequence) in enumerate(records, start=1):
        sequence = clean_sequence(raw_sequence)
        identifier = peptide_id(header) or f"pep_{index:06d}"
        if identifier in used_ids:
            identifier = f"{identifier}_{index:06d}"
        used_ids.add(identifier)

        failures = []
        if not (min_len <= len(sequence) <= max_len):
            failures.append("length_out_of_range")
        if any(residue not in STD_AA for residue in sequence):
            failures.append("non_standard_residue")
        if re.search(r"([A-Z])\1{%d,}" % (max_run - 1), sequence):
            failures.append("homopolymer_run")
        descriptors = all_descriptors(sequence) if sequence else None
        if descriptors is not None and not (charge_min <= descriptors["net_charge"] <= charge_max):
            failures.append("net_charge_out_of_range")
        if descriptors is not None and not (gravy_min <= descriptors["GRAVY"] <= gravy_max):
            failures.append("gravy_out_of_range")
        if sequence in exclusion:
            failures.append("matches_reference_peptide")
        if sequence in seen:
            failures.append("duplicate_in_input")

        if failures:
            for reason in set(failures):
                reasons[reason] += 1
            continue

        seen.add(sequence)
        passed.append((identifier, sequence))

    descriptor_rows = []
    for identifier, sequence in passed:
        row = {"id": identifier, "sequence": sequence}
        row.update(all_descriptors(sequence))
        descriptor_rows.append(row)

    columns = [
        "id", "sequence", "length", "MW", "pI", "net_charge", "GRAVY",
        "muH", "helix_prop", "cationic_ratio", "hydrophobic_ratio",
    ]
    csv_path = out_dir / "screened_physchem.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in descriptor_rows:
            writer.writerow({
                key: round(row[key], 4) if isinstance(row[key], float) else row[key]
                for key in columns
            })

    write_fasta(str(out_dir / "screened_physchem.fasta"), passed)

    summary = {
        "input_sequences": len(records),
        "passed_sequences": len(passed),
        "pass_rate": round(len(passed) / max(len(records), 1), 4),
        "rejected_sequences": len(records) - len(passed),
        "rejection_reasons": dict(reasons.most_common()),
        "note": "One sequence may trigger several rules; reason counts can exceed rejected sequences.",
        "parameters": params,
    }
    (out_dir / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Filtered: %d/%d passed (%.1f%%) -> %s",
        len(passed), len(records), 100 * len(passed) / max(len(records), 1), csv_path,
    )
    logger.info("Rejection reasons: %s", dict(reasons.most_common()))


if __name__ == "__main__":
    main()
