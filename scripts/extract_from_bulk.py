"""Stream the 8.5 GB CFPB bulk database and extract only the usable rows.

The full download is ~99% unusable for this project: most complaints carry no
narrative text, and credit-reporting bulk filings dominate the rest. Reading it
in chunks keeps memory flat regardless of file size.

Output feeds prepare_cfpb.py.

Usage:
    python extract_from_bulk.py --zip "C:/Users/namit/Downloads/complaints.csv.zip" --out cfpb_filtered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NARRATIVE = "Consumer complaint narrative"
CHUNK = 200_000

# Credit reporting is 81% of the database and has ~1.9M narratives post-2023.
# It is NOT excluded -- it is a legitimate product category and the largest one.
# It is only randomly downsampled here so the working pool stays a manageable
# size. The final per-product cap in prepare_cfpb.py is what actually balances
# the sample, and a random subset of a random subset is still random.
CR_PREFIX = "Credit reporting"
CR_POOL_FRACTION = 0.03


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", default=Path("cfpb_filtered.csv"), type=Path)
    ap.add_argument("--min-date", default="2023-01-01")
    ap.add_argument("--cr-fraction", type=float, default=CR_POOL_FRACTION,
                    help="fraction of credit-reporting rows to retain in the pool "
                         "(1.0 keeps all; 0 excludes the category entirely)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out.exists():
        args.out.unlink()

    scanned = kept = 0
    wrote_header = False

    reader = pd.read_csv(args.zip, chunksize=CHUNK, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        scanned += len(chunk)

        chunk = chunk[chunk[NARRATIVE].notna()]

        # The bulk file mixes "2023-04-11" and "2023-04-11T09:07:47Z" in the same column.
        dates = pd.to_datetime(chunk["Date received"], errors="coerce", utc=True, format="mixed")
        chunk = chunk[dates >= pd.Timestamp(args.min_date, tz="UTC")]

        # Downsample credit reporting only; every other product is kept whole.
        is_cr = chunk["Product"].str.startswith(CR_PREFIX, na=False)
        if args.cr_fraction < 1.0:
            cr = chunk[is_cr].sample(frac=args.cr_fraction, random_state=args.seed + i)
            chunk = pd.concat([chunk[~is_cr], cr])

        if len(chunk):
            chunk.to_csv(args.out, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            kept += len(chunk)

        if i % 10 == 0:
            print(f"  scanned {scanned:>10,}   kept {kept:>8,}", flush=True)

    print(f"\nscanned {scanned:,} rows -> kept {kept:,} ({kept/scanned:.2%})")
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1024**2:.0f} MB)")

    out = pd.read_csv(args.out, low_memory=False)
    print(f"\ndistinct issues: {out['Issue'].nunique()}")
    months = pd.to_datetime(out["Date received"], utc=True, format="mixed", errors="coerce")
    print(f"months covered:  {months.dt.to_period('M').nunique()}")
    print("\nproduct mix:")
    print(out["Product"].value_counts().to_string())


if __name__ == "__main__":
    main()
