"""Shared I/O, configuration and logging helpers.

The module intentionally depends only on the Python standard library and
PyYAML, so scripts that do not need deep-learning packages can still import it.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load a YAML configuration file into a dictionary."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return cfg


def ensure_dir(path: str) -> Path:
    """Create *path* (if needed) and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_sequence(text: str) -> str:
    """Keep alphabetic characters only and upper-case the result."""
    return re.sub(r"[^A-Za-z]", "", str(text or "")).upper()


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read a FASTA file as a list of ``(header, sequence)`` tuples.

    Multi-line records are supported. Blank lines are ignored and sequences
    are upper-cased.
    """
    records: List[Tuple[str, str]] = []
    header = None
    chunks: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    records.append((header, clean_sequence("".join(chunks))))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None and chunks:
        records.append((header, clean_sequence("".join(chunks))))
    return records


def write_fasta(path: str, records: Iterable[Tuple[str, str]]) -> None:
    """Write ``(header, sequence)`` tuples to a FASTA file."""
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def peptide_id(header: str) -> str:
    """Extract the identifier from a FASTA header (first whitespace token)."""
    return header.split()[0] if header.split() else header


def get_logger(name: str, logfile: str | None = None) -> logging.Logger:
    """Create a console logger, optionally mirrored to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if logfile:
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
