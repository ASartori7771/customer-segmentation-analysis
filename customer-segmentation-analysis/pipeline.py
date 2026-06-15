"""
Customer Data Ingestion Pipeline
---------------------------------
Reads raw CSV files from a folder, validates schema, cleans data,
merges multiple sources, and outputs a versioned clean artifact
with a full run summary.
"""

import os
import glob
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

RAW_DATA_DIR   = "./data/raw"          # folder with input CSVs
OUTPUT_DIR     = "./data/processed"    # where clean files are saved
LOG_DIR        = "./logs"              # where run logs are saved

# Expected columns and their expected dtypes (before coercion)
SCHEMA = {
    "CustomerID":                     "int",
    "Edad":                           "numeric",
    "Ingresos Anuales (k$)":          "numeric",
    "Puntuación de Gasto (1-100)":    "numeric",
    "Categoría de Producto Favorito": "string",
}

# Columns to impute with median if missing (instead of dropping the row)
IMPUTE_WITH_MEDIAN = [
    "Edad",
    "Ingresos Anuales (k$)",
    "Puntuación de Gasto (1-100)",
]

# Hard bounds for numeric columns: rows outside are treated as outliers
BOUNDS = {
    "Edad":                          (10, 100),
    "Ingresos Anuales (k$)":         (0,  1000),
    "Puntuación de Gasto (1-100)":   (1,  100),
}

# Valid values for categorical columns (None = accept anything)
VALID_CATEGORIES = {
    "Categoría de Producto Favorito": None   # set to a list to enforce, e.g. ["Electronics", "Books"]
}


# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────

def setup_logging(run_id: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")

    # Console handler — works in both terminal and Jupyter
    import sys
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"run_{run_id}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────
# STEP 1 — INGEST
# ──────────────────────────────────────────────

def ingest_csvs(folder: str, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """
    Scan a folder for CSV files and load each one into a dict keyed by filename.
    Logs a warning for empty files and skips them.
    """
    pattern = os.path.join(folder, "*.csv")
    paths   = sorted(glob.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No CSV files found in '{folder}'")

    logger.info(f"Found {len(paths)} CSV file(s) in '{folder}'")
    frames = {}

    for path in paths:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"  [SKIP] '{name}' is empty")
                continue
            logger.info(f"  [OK]   '{name}' — {len(df)} rows, {len(df.columns)} cols")
            frames[name] = df
        except Exception as e:
            logger.error(f"  [FAIL] Could not read '{name}': {e}")

    return frames


# ──────────────────────────────────────────────
# STEP 2 — VALIDATE SCHEMA
# ──────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, source: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Check that expected columns are present.
    Drops unrecognised columns with a warning.
    Raises if a required column is missing entirely.
    """
    missing = [col for col in SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(f"[{source}] Missing required columns: {missing}")

    extra = [col for col in df.columns if col not in SCHEMA]
    if extra:
        logger.warning(f"  [{source}] Dropping unrecognised columns: {extra}")
        df = df.drop(columns=extra)

    logger.info(f"  [{source}] Schema OK")
    return df


# ──────────────────────────────────────────────
# STEP 3 — CLEAN
# ──────────────────────────────────────────────

def clean(df: pd.DataFrame, source: str, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """
    Applies all cleaning steps and returns the cleaned DataFrame
    plus a per-source stats dict for the run summary.
    """
    stats = {
        "source":           source,
        "rows_in":          len(df),
        "duplicates_dropped": 0,
        "outliers_dropped":   0,
        "rows_imputed":       {},
        "rows_out":           0,
        "warnings":           [],
    }

    # ── 3a. Coerce numeric types ──────────────────────────────
    for col in IMPUTE_WITH_MEDIAN:
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced = df[col].isna().sum()
        if coerced > 0:
            msg = f"'{col}': {coerced} value(s) could not be parsed as numeric and were set to NaN"
            logger.warning(f"  [{source}] {msg}")
            stats["warnings"].append(msg)

    # ── 3b. Standardise strings ───────────────────────────────
    for col, valid in VALID_CATEGORIES.items():
        df[col] = df[col].astype(str).str.strip().str.title()
        if valid:
            bad_mask = ~df[col].isin(valid)
            if bad_mask.any():
                bad_vals = df.loc[bad_mask, col].unique().tolist()
                logger.warning(f"  [{source}] '{col}' has unrecognised values: {bad_vals} — setting to 'Unknown'")
                df.loc[bad_mask, col] = "Unknown"
                stats["warnings"].append(f"'{col}' unrecognised: {bad_vals}")

    # ── 3c. Deduplicate on CustomerID ─────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["CustomerID"], keep="first").copy()
    dropped = before - len(df)
    if dropped:
        logger.warning(f"  [{source}] Dropped {dropped} duplicate CustomerID row(s)")
    stats["duplicates_dropped"] = dropped

    # ── 3d. Impute missing numeric values with median ─────────
    for col in IMPUTE_WITH_MEDIAN:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  [{source}] Imputed {n_missing} missing '{col}' value(s) with median ({median_val:.2f})")
            stats["rows_imputed"][col] = int(n_missing)

    # ── 3e. Remove outliers ───────────────────────────────────
    outlier_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in BOUNDS.items():
        col_mask = (df[col] < lo) | (df[col] > hi)
        if col_mask.any():
            logger.warning(f"  [{source}] '{col}': {col_mask.sum()} outlier(s) outside [{lo}, {hi}] — dropping rows")
            outlier_mask |= col_mask
    df = df[~outlier_mask]
    stats["outliers_dropped"] = int(outlier_mask.sum())

    stats["rows_out"] = len(df)
    logger.info(
        f"  [{source}] Clean complete -- "
        f"{stats['rows_in']} in -> {stats['rows_out']} out "
        f"({stats['duplicates_dropped']} dupes, {stats['outliers_dropped']} outliers)"
    )
    return df, stats


# ──────────────────────────────────────────────
# STEP 4 — MERGE
# ──────────────────────────────────────────────

def merge_sources(frames: dict[str, pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """
    Concatenates all cleaned frames.
    If multiple files share the same CustomerID, keeps the first occurrence
    and logs a warning — in production you'd decide per-column which source wins.
    """
    if len(frames) == 1:
        return list(frames.values())[0]

    combined = pd.concat(frames.values(), ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=["CustomerID"], keep="first")
    after    = len(combined)

    if before != after:
        logger.warning(
            f"  [MERGE] {before - after} CustomerID collision(s) across files — kept first occurrence"
        )

    logger.info(f"  [MERGE] {len(frames)} source(s) merged -> {after} total rows")
    return combined


# ──────────────────────────────────────────────
# STEP 5 — SAVE
# ──────────────────────────────────────────────

def save_output(df: pd.DataFrame, run_id: str, logger: logging.Logger) -> str:
    """
    Saves the clean DataFrame as a versioned CSV in OUTPUT_DIR.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"customers_clean_{run_id}.csv"
    path     = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"  [SAVE] Clean data written to '{path}'")
    return path


# ──────────────────────────────────────────────
# STEP 6 — RUN SUMMARY
# ──────────────────────────────────────────────

def save_summary(all_stats: list[dict], run_id: str, output_path: str, logger: logging.Logger):
    """
    Writes a JSON run summary to the log directory.
    """
    total_in  = sum(s["rows_in"]  for s in all_stats)
    total_out = sum(s["rows_out"] for s in all_stats)

    summary = {
        "run_id":       run_id,
        "output_file":  output_path,
        "total_rows_in":  total_in,
        "total_rows_out": total_out,
        "rows_dropped":   total_in - total_out,
        "sources":        all_stats,
    }

    summary_path = os.path.join(LOG_DIR, f"summary_{run_id}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("-" * 55)
    logger.info("RUN SUMMARY")
    logger.info(f"  Run ID       : {run_id}")
    logger.info(f"  Rows in      : {total_in}")
    logger.info(f"  Rows out     : {total_out}")
    logger.info(f"  Rows dropped : {total_in - total_out}")
    for s in all_stats:
        logger.info(
            f"  [{s['source']}]  dupes={s['duplicates_dropped']}  "
            f"outliers={s['outliers_dropped']}  imputed={s['rows_imputed']}"
        )
    if any(s["warnings"] for s in all_stats):
        logger.warning("  Warnings:")
        for s in all_stats:
            for w in s["warnings"]:
                logger.warning(f"    [{s['source']}] {w}")
    logger.info(f"  Full summary  : {summary_path}")
    logger.info("-" * 55)


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(run_id)

    logger.info("=" * 55)
    logger.info(f"PIPELINE START — run_id={run_id}")
    logger.info("=" * 55)

    # 1. Ingest
    raw_frames = ingest_csvs(RAW_DATA_DIR, logger)

    # 2. Validate + 3. Clean each source independently
    clean_frames = {}
    all_stats    = []

    for name, df in raw_frames.items():
        logger.info(f"Processing '{name}' ...")
        df = validate_schema(df, name, logger)
        df, stats = clean(df, name, logger)
        clean_frames[name] = df
        all_stats.append(stats)

    # 4. Merge
    logger.info("Merging sources ...")
    final_df = merge_sources(clean_frames, logger)

    # 5. Save
    output_path = save_output(final_df, run_id, logger)

    # 6. Summary
    save_summary(all_stats, run_id, output_path, logger)

    logger.info("PIPELINE COMPLETE")
    return final_df


if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())