"""Merge RFMiD, ODIR and DeepDRiD CSV manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def read_rfmid(path: Path) -> pd.DataFrame:
    """Return dataframe for the RFMiD labels."""
    df = pd.read_csv(path)
    df["source"] = "rfmid"
    return df


def read_odir(path: Path) -> pd.DataFrame:
    """Return dataframe for the ODIR-5K labels."""
    df = pd.read_csv(path)
    # try common column names for image identifier
    for col in ["image", "image_path", "img", "filename"]:
        if col in df.columns:
            df["ID"] = df[col].apply(lambda p: Path(p).stem)
            df = df.drop(columns=[col])
            break
    df["source"] = "odir"
    return df


def read_deepdrid(path: Path) -> pd.DataFrame:
    """Return dataframe for the DeepDRiD labels."""
    df = pd.read_csv(path)
    if "image" in df.columns:
        df["ID"] = df["image"].apply(lambda p: Path(p).stem)
        df = df.drop(columns=["image"])
    df["source"] = "deepdrid"
    return df


def harmonise_columns(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Ensure all dataframes share the same column set."""
    columns = set()
    for df in dfs:
        columns.update(df.columns)
    # ensure a deterministic order and ID appears first
    ordered = ["ID"] + sorted(c for c in columns if c != "ID")
    for i, df in enumerate(dfs):
        for col in ordered:
            if col not in df.columns:
                df[col] = ""
        dfs[i] = df[ordered]
    return dfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine dataset CSVs")
    parser.add_argument("rfmid", type=Path, help="Path to rfmid.csv")
    parser.add_argument("odir", type=Path, help="Path to full_df.csv from ODIR-5K")
    parser.add_argument("deepdrid", type=Path, help="Path to deepdrid.csv")
    parser.add_argument("--output", type=Path, default=Path("combined_dataset.csv"), help="Output CSV filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rfmid = read_rfmid(args.rfmid)
    odir = read_odir(args.odir)
    deepdrid = read_deepdrid(args.deepdrid)

    rfmid, odir, deepdrid = harmonise_columns([rfmid, odir, deepdrid])
    combined = pd.concat([rfmid, odir, deepdrid], ignore_index=True)
    combined.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
