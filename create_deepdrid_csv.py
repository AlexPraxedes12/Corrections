"""Generate a CSV manifest for the DeepDRiD dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


GRADE_MAP = {
    "normal": 0,
    "no_dr": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
    "pdr": 4,
    "proliferative": 4,
}


def infer_grade_from_path(path: Path) -> Optional[int]:
    """Return the diabetic retinopathy grade inferred from ``path``."""
    tokens = "_".join(path.parts).lower()
    match = re.search(r"grade[_-]?([0-4])", tokens)
    if match:
        return int(match.group(1))

    match = re.search(r"dr[_-]?([0-4])", tokens)
    if match:
        return int(match.group(1))

    for key, val in GRADE_MAP.items():
        if key in tokens:
            return val

    match = re.search(r"(?:^|[_-])([0-4])(?:[_-]|$)", path.stem)
    if match:
        return int(match.group(1))

    return None


def scan_images(directory: Path) -> List[Dict[str, str | int]]:
    """Return rows describing images found in ``directory``."""
    rows: List[Dict[str, str | int]] = []
    for img_path in directory.rglob("*"):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        grade = infer_grade_from_path(img_path)
        rows.append(
            {
                "image": str(img_path.relative_to(directory.parent)),
                "DR_grade": grade if grade is not None else "",
                "modality": directory.name,
            }
        )
    return rows


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    """Collect image metadata from ``dataset_root``."""
    rows: List[Dict[str, str | int]] = []
    for sub in ["regular_fundus_images", "ultra-widefield_images"]:
        subdir = dataset_root / sub
        if subdir.exists():
            rows.extend(scan_images(subdir))
    return pd.DataFrame(rows, columns=["image", "DR_grade", "modality"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DeepDRiD CSV manifest")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the DeepDRiD dataset root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deepdrid.csv",
        help="Filename of the generated CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(Path(args.data_path))
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
