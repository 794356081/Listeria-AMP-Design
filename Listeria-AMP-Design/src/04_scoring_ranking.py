#!/usr/bin/env python
"""Five-component scoring and ranking of candidate peptides.

Inputs
------
``--physchem``
    Descriptor CSV produced by ``02_physicochemical_filter.py``.
``--embeddings`` / ``--embedding-index``
    Candidate ESM-2 matrix and index produced by ``03_esm2_embedding.py``.
``--known-amps``
    Reference peptides with columns ``peptide_id, sequence, mic_ug_ml, hemolytic``.
    ``hemolytic`` is 1 (hemolytic), 0 (non-hemolytic) or empty (unknown).
``--known-embeddings`` / ``--known-embedding-index``
    ESM-2 matrix and index for the reference peptides.

Scoring
-------
W1  ESM-2 similarity to the high-activity reference centroid   weight 0.45
W2  cationic residue ratio and hydrophobic moment               weight 0.20
W3  Chou-Fasman helix propensity                                weight 0.15
W4  hydrophobicity balance (Gaussian around GRAVY = 0.2)        weight 0.10
W5  low hemolysis risk                                          weight 0.10

W1-W3 are percentile ranks within the scored library. W5 uses a logistic
regression trained on reference peptides with hemolysis labels when enough
labels are available; otherwise a physicochemical heuristic is used.

Outputs
-------
``<out-dir>/scoring_result.csv``   all candidates, sorted by score
``<out-dir>/top_candidates.csv``   top ``--top-n`` candidates
``<out-dir>/scoring_summary.json`` run metadata and score statistics

Example
-------
python src/04_scoring_ranking.py \
    --physchem runs/02_filter/screened_physchem.csv \
    --embeddings runs/03_embedding/esm2_embeddings.npy \
    --embedding-index runs/03_embedding/esm2_embedding_index.csv \
    --known-amps data/known_amps_example.csv \
    --known-embeddings examples/example_known_embeddings.npy \
    --known-embedding-index examples/example_known_embedding_index.csv \
    --out-dir runs/04_scoring
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import rankdata

from common import ensure_dir, get_logger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--physchem", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-index", required=True)
    parser.add_argument("--known-amps", required=True)
    parser.add_argument("--known-embeddings", required=True)
    parser.add_argument("--known-embedding-index", required=True)
    parser.add_argument("--out-dir", default="runs/04_scoring")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] using average ranks."""
    values = np.asarray(values, dtype=float)
    return (rankdata(values, method="average") - 1) / max(len(values) - 1, 1)


def read_csv_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_index(path: str) -> dict[str, int]:
    """Map peptide id and sequence to matrix row index."""
    mapping = {}
    for row in read_csv_rows(path):
        row_index = int(row["row_index"])
        mapping[("id", row["peptide_id"])] = row_index
        mapping[("seq", row["sequence"])] = row_index
    return mapping


def parse_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_label(value):
    value = str(value or "").strip().lower()
    if value in ("1", "true", "yes", "hemolytic"):
        return 1
    if value in ("0", "false", "no", "non-hemolytic", "nonhemolytic"):
        return 0
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    scoring_cfg = cfg.get("scoring", {})
    weights = scoring_cfg.get("weights", {})
    w1_weight = float(weights.get("esm_similarity", 0.45))
    w2_weight = float(weights.get("cationic_amphipathic", 0.20))
    w3_weight = float(weights.get("helix_propensity", 0.15))
    w4_weight = float(weights.get("hydrophobicity_balance", 0.10))
    w5_weight = float(weights.get("low_hemolysis", 0.10))
    target_gravy = float(scoring_cfg.get("hydrophobicity_balance", {}).get("target_gravy", 0.2))
    sigma_gravy = float(scoring_cfg.get("hydrophobicity_balance", {}).get("sigma", 0.5))
    quantile = float(scoring_cfg.get("high_activity_quantile", 0.5))
    hemo_cfg = scoring_cfg.get("hemolysis_model", {})

    out_dir = ensure_dir(args.out_dir)
    logger = get_logger("scoring", str(out_dir / "scoring.log"))

    candidates = read_csv_rows(args.physchem)
    if not candidates:
        raise SystemExit("The physicochemical table is empty")
    candidate_index = read_index(args.embedding_index)
    candidate_matrix = np.load(args.embeddings, mmap_mode="r")

    candidate_vectors = []
    kept = []
    for row in candidates:
        row_index = candidate_index.get(("id", row["id"]))
        if row_index is None:
            row_index = candidate_index.get(("seq", row["sequence"]))
        if row_index is None:
            raise SystemExit(f"No embedding found for candidate {row['id']}")
        vector = candidate_matrix[row_index]
        if np.isnan(vector).any():
            logger.warning("Dropping candidate %s: embedding contains NaN", row["id"])
            continue
        candidate_vectors.append(vector)
        kept.append(row)
    candidates = kept
    if not candidates:
        raise SystemExit("No candidate with a valid embedding remains")
    X = np.vstack(candidate_vectors).astype(np.float32)
    logger.info("Scoring %d candidates with %d-dimensional embeddings", X.shape[0], X.shape[1])

    known_rows = read_csv_rows(args.known_amps)
    known_index = read_index(args.known_embedding_index)
    known_matrix = np.load(args.known_embeddings, mmap_mode="r")
    known = []
    known_vectors = []
    for row in known_rows:
        row_index = known_index.get(("id", row["peptide_id"]))
        if row_index is None:
            row_index = known_index.get(("seq", row["sequence"]))
        if row_index is None:
            continue
        vector = known_matrix[row_index]
        if np.isnan(vector).any():
            continue
        row["mic"] = parse_float(row.get("mic_ug_ml"))
        row["hemolytic"] = parse_label(row.get("hemolytic"))
        known.append(row)
        known_vectors.append(vector)
    if not known:
        raise SystemExit("No reference peptide with a valid embedding was found")
    Xo = np.vstack(known_vectors).astype(np.float32)
    logger.info("Reference set: %d peptides (%d with MIC, %d with hemolysis label)",
                len(known), sum(1 for r in known if r["mic"] is not None),
                sum(1 for r in known if r["hemolytic"] is not None))

    mic_values = np.array([r["mic"] for r in known if r["mic"] is not None], dtype=float)
    if mic_values.size == 0:
        raise SystemExit("At least one reference MIC value is required for W1")
    threshold = float(np.quantile(mic_values, quantile))
    reference_indices = [index for index, row in enumerate(known)
                         if row["mic"] is not None and row["mic"] <= threshold]
    if not reference_indices:
        raise SystemExit("The high-activity reference set is empty")

    mu = Xo.mean(axis=0)
    Xc = X - mu
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    V = Xc / np.maximum(norms, 1e-12)
    Vo = Xo - mu
    Vo = Vo / np.maximum(np.linalg.norm(Vo, axis=1, keepdims=True), 1e-12)
    centroid = Vo[reference_indices].mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    similarity = V @ centroid

    w1 = percentile_rank(similarity)
    mu_h = np.array([float(r["muH"]) for r in candidates])
    cationic = np.array([float(r["cationic_ratio"]) for r in candidates])
    w2 = 0.5 * percentile_rank(cationic) + 0.5 * percentile_rank(mu_h)
    w3 = percentile_rank(np.array([float(r["helix_prop"]) for r in candidates]))
    gravy = np.array([float(r["GRAVY"]) for r in candidates])
    w4 = np.exp(-((gravy - target_gravy) / sigma_gravy) ** 2)

    labels = [(row, vector) for row, vector in zip(known, known_vectors)
              if row["hemolytic"] is not None]
    y = np.array([row["hemolytic"] for row, _ in labels], dtype=int)
    hemo_auc = None
    hemo_method = "heuristic"
    hemo_probability = None
    min_labels = int(hemo_cfg.get("min_labels", 16))
    min_auc = float(hemo_cfg.get("min_cv_auc", 0.60))
    if len(y) >= min_labels and len(set(y.tolist())) == 2:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler

            Xl = np.vstack([vector for _, vector in labels]).astype(np.float32)
            scaler = StandardScaler().fit(Xl)
            Xls = scaler.transform(Xl)
            folds = StratifiedKFold(
                n_splits=int(hemo_cfg.get("n_splits", 5)), shuffle=True, random_state=0
            )
            aucs = []
            for train, test in folds.split(Xls, y):
                model = LogisticRegression(
                    C=float(hemo_cfg.get("C", 0.1)),
                    max_iter=int(hemo_cfg.get("max_iter", 3000)),
                    class_weight="balanced",
                ).fit(Xls[train], y[train])
                try:
                    aucs.append(roc_auc_score(y[test], model.predict_proba(Xls[test])[:, 1]))
                except ValueError:
                    continue
            hemo_auc = float(np.mean(aucs)) if aucs else 0.0
            if hemo_auc >= min_auc:
                model = LogisticRegression(
                    C=float(hemo_cfg.get("C", 0.1)),
                    max_iter=int(hemo_cfg.get("max_iter", 3000)),
                    class_weight="balanced",
                ).fit(Xls, y)
                hemo_probability = model.predict_proba(scaler.transform(X))[:, 1]
                hemo_method = "logistic_regression_esm2"
        except ImportError:
            logger.warning("scikit-learn is unavailable; using the heuristic hemolysis score")

    if hemo_probability is None:
        hemo_probability = (
            0.40 * percentile_rank(mu_h)
            + 0.25 * percentile_rank(gravy)
            + 0.20 * percentile_rank(np.array([float(r["net_charge"]) for r in candidates]))
            + 0.15 * percentile_rank(np.array([float(r["hydrophobic_ratio"]) for r in candidates]))
        )
    w5 = 1.0 - np.clip(hemo_probability, 0.0, 1.0)

    score = 100.0 * (
        w1_weight * w1 + w2_weight * w2 + w3_weight * w3 + w4_weight * w4 + w5_weight * w5
    )
    order = np.argsort(-score, kind="mergesort")

    columns = [
        "rank", "peptide_id", "sequence", "length", "MW", "pI", "net_charge",
        "GRAVY", "muH", "helix_prop", "cationic_ratio", "hemolysis_prob",
        "W1_esm_similarity", "W2_cationic_amphipathic", "W3_helix_propensity",
        "W4_hydrophobicity_balance", "W5_low_hemolysis", "score",
    ]
    result_rows = []
    for rank, index in enumerate(order, start=1):
        row = candidates[index]
        result_rows.append({
            "rank": rank,
            "peptide_id": row["id"],
            "sequence": row["sequence"],
            "length": int(row["length"]),
            "MW": round(float(row["MW"]), 2),
            "pI": round(float(row["pI"]), 2),
            "net_charge": round(float(row["net_charge"]), 3),
            "GRAVY": round(float(row["GRAVY"]), 4),
            "muH": round(float(row["muH"]), 4),
            "helix_prop": round(float(row["helix_prop"]), 4),
            "cationic_ratio": round(float(row["cationic_ratio"]), 4),
            "hemolysis_prob": round(float(hemo_probability[index]), 4),
            "W1_esm_similarity": round(float(w1[index]), 4),
            "W2_cationic_amphipathic": round(float(w2[index]), 4),
            "W3_helix_propensity": round(float(w3[index]), 4),
            "W4_hydrophobicity_balance": round(float(w4[index]), 4),
            "W5_low_hemolysis": round(float(w5[index]), 4),
            "score": round(float(score[index]), 3),
        })

    result_path = out_dir / "scoring_result.csv"
    with open(result_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_rows)

    top_path = out_dir / "top_candidates.csv"
    with open(top_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_rows[: args.top_n])

    summary = {
        "candidates_scored": len(result_rows),
        "reference_peptides": len(known),
        "mic_median": round(float(np.median(mic_values)), 4),
        "high_activity_threshold": round(threshold, 4),
        "high_activity_reference_count": len(reference_indices),
        "hemolysis_model": hemo_method,
        "hemolysis_cv_auc": None if hemo_auc is None else round(hemo_auc, 4),
        "weights": {
            "W1_esm_similarity": w1_weight,
            "W2_cationic_amphipathic": w2_weight,
            "W3_helix_propensity": w3_weight,
            "W4_hydrophobicity_balance": w4_weight,
            "W5_low_hemolysis": w5_weight,
        },
        "score_min": round(float(score.min()), 3),
        "score_max": round(float(score.max()), 3),
        "score_mean": round(float(score.mean()), 3),
        "note": "Scores are relative to the scored library; percentile ranks change with the input set.",
    }
    (out_dir / "scoring_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Finished: %d candidates | score %.2f-%.2f | %s",
                len(result_rows), score.min(), score.max(), result_path)


if __name__ == "__main__":
    main()
