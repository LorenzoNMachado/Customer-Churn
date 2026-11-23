"""Download and verify the pinned Kaggle Telco Customer Churn CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
EXPECTED_ROWS = 7_043
EXPECTED_HEADER = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / FILENAME


class DownloadUnavailableError(Exception):
    """The Kaggle download endpoint could not be reached or did not allow access."""


def fetch_csv() -> bytes:
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Customer-Churn-research"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            archive = response.read()
    except urllib.error.HTTPError as error:
        raise DownloadUnavailableError(
            f"Kaggle returned HTTP {error.code}. Download {FILENAME} manually from "
            f"{DOWNLOAD_URL.replace('/api/v1/datasets/download/', '/datasets/')} and "
            "save it under data/."
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise DownloadUnavailableError(
            f"Could not download the Kaggle dataset ({error}). Download {FILENAME} "
            "manually from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and save it under data/."
        ) from error

    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = [name for name in zipped.namelist() if Path(name).name == FILENAME]
        if len(names) != 1:
            raise ValueError(f"Expected one {FILENAME} in the Kaggle archive; found {len(names)}.")
        content = zipped.read(names[0])

    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset SHA-256 changed: expected {EXPECTED_SHA256}, got {digest}.")

    rows = csv.reader(io.TextIOWrapper(io.BytesIO(content), encoding="utf-8-sig", newline=""))
    if next(rows, None) != EXPECTED_HEADER:
        raise ValueError("Dataset columns differ from the pinned source version.")
    row_count = sum(1 for _ in rows)
    if row_count != EXPECTED_ROWS:
        raise ValueError(f"Dataset row count differs: expected {EXPECTED_ROWS}, got {row_count}.")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(f"{args.output} already exists; choose another --output or pass --force.")

    content = fetch_csv()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        if args.force:
            temporary_path.replace(args.output)
        else:
            os.link(temporary_path, args.output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Saved verified dataset to {args.output} (SHA-256 {EXPECTED_SHA256}).")


if __name__ == "__main__":
    try:
        main()
    except DownloadUnavailableError as error:
        print(f"Dataset download unavailable: {error}", file=sys.stderr)
        sys.exit(3)
    except (zipfile.BadZipFile, UnicodeError, csv.Error, OSError, ValueError) as error:
        print(f"Dataset download or validation failed: {error}", file=sys.stderr)
        sys.exit(1)
