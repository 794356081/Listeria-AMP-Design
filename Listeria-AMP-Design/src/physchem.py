"""Physicochemical descriptors used by the filtering and scoring stages.

All descriptors are sequence-based. The residue scales are conventional
published parameter sets (see docs/parameter_details.md).
"""
from __future__ import annotations

import math
from typing import Dict

RESIDUE_MASS = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}

# Kyte-Doolittle hydropathy scale (GRAVY).
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# Eisenberg consensus hydrophobicity scale (hydrophobic moment).
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85,
    "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50,
    "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81,
    "Y": 0.26, "V": 1.08,
}

# Chou-Fasman alpha-helix propensity (Pa).
HELIX_PROPENSITY = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11,
    "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08, "L": 1.21, "K": 1.16,
    "M": 1.45, "F": 1.13, "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08,
    "Y": 0.69, "V": 1.06,
}

PKA = {
    "Nterm": 9.69, "Cterm": 2.34, "D": 3.65, "E": 4.25, "C": 8.18,
    "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48,
}

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def molecular_weight(sequence: str) -> float:
    """Monoisotopic residue-mass sum plus one water molecule."""
    return sum(RESIDUE_MASS[a] for a in sequence if a in RESIDUE_MASS) + 18.01528


def net_charge(sequence: str, ph: float = 7.4) -> float:
    """Henderson-Hasselbalch net charge at a given pH."""
    charge = 1.0 / (1.0 + 10 ** (ph - PKA["Nterm"])) - 1.0 / (
        1.0 + 10 ** (PKA["Cterm"] - ph)
    )
    for residue in sequence:
        if residue in ("K", "R", "H"):
            charge += 1.0 / (1.0 + 10 ** (ph - PKA[residue]))
        elif residue in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (PKA[residue] - ph))
    return charge


def isoelectric_point(sequence: str) -> float:
    """Bisection estimate of the pH at which the net charge is zero."""
    low, high = 0.0, 14.0
    for _ in range(50):
        middle = (low + high) / 2
        if net_charge(sequence, middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def gravy(sequence: str) -> float:
    """Grand average of hydropathy (Kyte-Doolittle scale)."""
    values = [KD_HYDROPATHY[a] for a in sequence if a in KD_HYDROPATHY]
    return sum(values) / len(values) if values else 0.0


def hydrophobic_moment(sequence: str, window: int = 11, angle: float = 100.0) -> float:
    """Maximum hydrophobic moment over a sliding window.

    The window is truncated to the peptide length for short sequences.
    ``angle`` is the residue rotation per turn of an ideal alpha-helix.
    """
    length = len(sequence)
    if length == 0:
        return 0.0
    width = min(window, length)
    radians = math.radians(angle)
    best = 0.0
    for start in range(length - width + 1):
        window_seq = sequence[start:start + width]
        real = sum(
            EISENBERG[a] * math.cos(i * radians)
            for i, a in enumerate(window_seq)
            if a in EISENBERG
        )
        imag = sum(
            EISENBERG[a] * math.sin(i * radians)
            for i, a in enumerate(window_seq)
            if a in EISENBERG
        )
        best = max(best, math.hypot(real, imag) / width)
    return best


def helix_propensity(sequence: str) -> float:
    """Mean Chou-Fasman alpha-helix propensity."""
    values = [HELIX_PROPENSITY[a] for a in sequence if a in HELIX_PROPENSITY]
    return sum(values) / len(values) if values else 0.0


def cationic_ratio(sequence: str) -> float:
    """Fraction of lysine and arginine residues."""
    return sum(1 for a in sequence if a in ("K", "R")) / max(1, len(sequence))


def hydrophobic_ratio(sequence: str) -> float:
    """Fraction of A, V, I, L, M, F, W and Y residues."""
    return sum(1 for a in sequence if a in "AVILMFWY") / max(1, len(sequence))


def all_descriptors(sequence: str) -> Dict[str, float]:
    """Compute the complete descriptor set used by the pipeline."""
    return {
        "length": len(sequence),
        "MW": molecular_weight(sequence),
        "pI": isoelectric_point(sequence),
        "net_charge": net_charge(sequence),
        "GRAVY": gravy(sequence),
        "muH": hydrophobic_moment(sequence),
        "helix_prop": helix_propensity(sequence),
        "cationic_ratio": cationic_ratio(sequence),
        "hydrophobic_ratio": hydrophobic_ratio(sequence),
    }
