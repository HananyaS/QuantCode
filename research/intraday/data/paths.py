"""Default cache roots (main checkout, gitignored) and cache file paths."""
from __future__ import annotations

from datetime import date
from pathlib import Path

DEFAULT_DATA_ROOT: Path = Path(
    "C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_penny")
DEFAULT_VOLUME_DAILY_ROOT: Path = Path(
    "C:/Users/Shachar/Desktop/PythonProjects/QuantCode/data/research_breakout/daily")


def _day_dir(root: Path, kind: str, d: date) -> Path:
    return root / kind / f"dt={d:%Y-%m-%d}"


def minute_file(root: Path, d: date) -> Path:
    return _day_dir(root, "minute", d) / "minute.parquet"


def requested_file(root: Path, d: date) -> Path:
    return _day_dir(root, "minute", d) / "requested.json"


def premarket_file(root: Path, d: date) -> Path:
    return _day_dir(root, "premarket", d) / "premarket.parquet"


def daily_file(root: Path, year: int) -> Path:
    return root / "daily" / f"bars_{year}.parquet"


def volume_daily_file(volume_root: Path, year: int) -> Path:
    return volume_root / f"bars_{year}.parquet"


def quote_summary_files(root: Path) -> tuple[Path, Path]:
    return root / "quote_summary_r4.parquet", root / "quote_summary_rev.parquet"
