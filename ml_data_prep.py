"""
preprocess_farmer_data.py
────────────────────────────────
GrowForMe Credit Scoring — Feature Engineering & Preprocessing Pipeline

Transforms raw farmer input data into an ML-ready feature matrix.

INPUT  : CSV or Excel file containing raw farmer records (one row per farmer)
OUTPUT : CSV file ready for model training or inference

Required raw columns
────────────────────
    farmer_id, gender, drought_flood_index, savings_ghs, payment_frequency,
    acres, satellite_verified, repayment_rate, yield_data,
    endorsements, irrigation_type, irrigation_scheme, market_access_index,
    training_sessions, livestock_value_ghs, alternative_income_ghs,
    insurance_type, insurance_subscription, digital_score, soil_health_index,
    farmer_budget_ghs, crop_types

Optional (present in training data, not required for inference)
    credit_score, creditworthiness

Usage
─────
    # Basic — reads CSV, writes ML-ready CSV next to it
    python preprocess_farmer_data.py --input data/farmers.csv

    # Specify output path
    python preprocess_farmer_data.py --input data/farmers.csv --output data/farmers_ml.csv

    # From Excel workbook (reads the '🌾 Farmer Data' sheet by default)
    python preprocess_farmer_data.py --input data/farmers.xlsx --sheet "Farmer Data"

    # Drop credit_score / creditworthiness for inference (no labels available)
    python preprocess_farmer_data.py --input data/farmers.csv --inference

    # Preview first 5 rows without saving
    python preprocess_farmer_data.py --input data/farmers.csv --preview

Dependencies
────────────
    pip install pandas openpyxl
"""

import argparse
import json
import sys
import os
import pandas as pd


#  CONSTANTS

CREDITWORTHINESS_CODE = {"Excellent": 3, "Good": 2, "Fair": 1, "Poor": 0}
FX_RATE_GHS_PER_USD = 15.0

# Base columns expected in raw input (money fields are handled flexibly below)
REQUIRED_COLS = [
    "farmer_id", "gender",
    "drought_flood_index", "payment_frequency",
    "acres", "satellite_verified", "repayment_rate", "yield_data",
    "endorsements", "irrigation_type", "irrigation_scheme",
    "market_access_index", "training_sessions",
    "is_association_member", "has_motorbike",
    "insurance_type", "insurance_subscription",
    "digital_score", "soil_health_index",
    "farmer_budget_ghs", "crop_types",
]

MONEY_FEATURES = [
    ("savings_usd", "savings_ghs"),
    ("livestock_value_usd", "livestock_value_ghs"),
    ("alternative_income_usd", "alternative_income_ghs"),
]

# Columns only present in labelled (training) data
LABEL_COLS = ["credit_score", "creditworthiness"]

# Final ordered feature columns produced by this script
# X = all columns except farmer_id, credit_score, creditworthiness_code
# y = credit_score  (regression target, continuous 0–100)
# y_cat = creditworthiness_code  (ordinal: Excellent=3, Good=2, Fair=1, Poor=0)
OUTPUT_COLS = [
    # identifier 
    "farmer_id",

    # continuous numeric features 
    "drought_flood_index",      # environmental risk 0–100
    "savings_usd",              # mobile money savings (USD)
    "payment_frequency",        # annual mobile money transactions
    "acres",                    # farm size
    "repayment_rate",           # prior loan repayment % (0–100)
    "yield_avg",                # mean of 3 seasonal yields (derived)
    "farmer_budget_ghs",        # seasonal net budget (GHS, can be negative)
    "endorsements",             # peer endorsement count
    "market_access_index",      # GIS market proximity 0–100
    "training_sessions",        # agricultural training sessions attended
    "livestock_value_usd",      # livestock value (USD)
    "alternative_income_usd",   # non-farm monthly income (USD)
    "digital_score",            # digital engagement 0–100
    "soil_health_index",        # soil quality 0–100

    # binary features (0/1) 
    "gender_female",            # 1 = female, 0 = male
    "is_association_member",    # cooperative membership
    "has_motorbike",            # motorbike ownership
    "satellite_verified",       # farm size satellite-verified
    "irrigation_scheme",        # formal GIDA irrigation scheme
    "insurance_subscription",   # active insurance premium payments

    # one-hot: irrigation_type (drip / canal / none) 
    "irr_none",
    "irr_canal",
    "irr_drip",

    # one-hot: insurance_type (none / crop / livestock / both) ─
    "ins_none",
    "ins_crop",
    "ins_livestock",
    "ins_both",

    # multi-hot: crop_types components 
    "has_staple",               # maize, yam, cassava etc.
    "has_cash_crop",            # cocoa, oil palm etc.
    "has_vegetable",            # tomato, pepper etc.
    "has_other",

    # targets (present in training data only) 
    "credit_score",             # continuous 0–100  ← regression target (y)
    "creditworthiness_code",    # ordinal 0–3       ← classification target (optional)
]


#  HELPER: normalise boolean-like values from various source formats

def to_bool_int(val) -> int:
    """
    Converts TRUE/FALSE strings, Python bools, 1/0 integers, or yes/no
    into a clean 0 or 1 integer.
    """
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(bool(val))
    s = str(val).strip().lower()
    if s in ("true", "yes", "1", "t", "y"):
        return 1
    if s in ("false", "no", "0", "f", "n"):
        return 0
    raise ValueError(f"Cannot convert '{val}' to boolean integer")


#  CORE TRANSFORMATION

def preprocess(df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
    """
    Transforms a raw farmer DataFrame into the ML-ready feature matrix.

    Parameters
    ----------
    df        : Raw input DataFrame (one row per farmer).
    inference : If True, skips label columns (credit_score / creditworthiness).
                Use this when preprocessing new farmers for scoring, not training.

    Returns
    -------
    pd.DataFrame with columns matching OUTPUT_COLS (minus label columns if inference=True).
    """

    # 0. Validate required columns 
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing {len(missing)} required column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    money_missing = []
    for usd_col, ghs_col in MONEY_FEATURES:
        if usd_col not in df.columns and ghs_col not in df.columns:
            money_missing.append(f"{usd_col} (or {ghs_col})")
    if money_missing:
        raise ValueError(
            f"Input data is missing money columns: {money_missing}.\n"
            "Provide USD directly or GHS so pipeline can convert using fx rate."
        )

    def money_to_usd(frame: pd.DataFrame, usd_col: str, ghs_col: str) -> pd.Series:
        if usd_col in frame.columns:
            return pd.to_numeric(frame[usd_col], errors="coerce").clip(lower=0)
        return (pd.to_numeric(frame[ghs_col], errors="coerce") / FX_RATE_GHS_PER_USD).clip(lower=0)

    out = pd.DataFrame()

    # 1. Pass-through identifier
    out["farmer_id"] = df["farmer_id"]

    # 2. Continuous numeric features — cast and clip to valid ranges ───────
    out["drought_flood_index"]    = pd.to_numeric(df["drought_flood_index"],    errors="coerce").clip(0, 100)
    out["savings_usd"]            = money_to_usd(df, "savings_usd", "savings_ghs")
    out["payment_frequency"]      = pd.to_numeric(df["payment_frequency"],      errors="coerce").clip(0, 52).round().astype("Int64")
    out["acres"]                  = pd.to_numeric(df["acres"],                  errors="coerce").clip(lower=0)
    out["repayment_rate"]         = pd.to_numeric(df["repayment_rate"],         errors="coerce").clip(0, 100)
    out["farmer_budget_ghs"]      = pd.to_numeric(df["farmer_budget_ghs"],      errors="coerce")   # can be negative
    out["endorsements"]           = pd.to_numeric(df["endorsements"],           errors="coerce").clip(lower=0).round().astype("Int64")
    out["market_access_index"]    = pd.to_numeric(df["market_access_index"],    errors="coerce").clip(0, 100)
    out["training_sessions"]      = pd.to_numeric(df["training_sessions"],      errors="coerce").clip(lower=0).round().astype("Int64")
    out["livestock_value_usd"]    = money_to_usd(df, "livestock_value_usd", "livestock_value_ghs")
    out["alternative_income_usd"] = money_to_usd(df, "alternative_income_usd", "alternative_income_ghs")
    out["digital_score"]          = pd.to_numeric(df["digital_score"],          errors="coerce").clip(0, 100)
    out["soil_health_index"]      = pd.to_numeric(df["soil_health_index"],      errors="coerce").clip(0, 100)

    # 3. yield_data → yield_avg─
    # Raw column is a comma-separated string of 3 seasonal yield values,
    # e.g. "102.3,98.7,115.0"  (each relative to regional average = 100)
    def parse_yield_avg(val) -> float:
        try:
            parts = [float(x.strip()) for x in str(val).split(",") if x.strip()]
            return round(sum(parts) / len(parts), 2) if parts else float("nan")
        except (ValueError, ZeroDivisionError):
            return float("nan")

    out["yield_avg"] = df["yield_data"].apply(parse_yield_avg).clip(lower=0)

    # 4. Binary features ──────
    for raw_col, out_col in [
        ("is_association_member", "is_association_member"),
        ("has_motorbike",         "has_motorbike"),
        ("satellite_verified",    "satellite_verified"),
        ("irrigation_scheme",     "irrigation_scheme"),
        ("insurance_subscription","insurance_subscription"),
    ]:
        out[out_col] = df[raw_col].apply(to_bool_int)

    # gender → gender_female (1 = female, 0 = male/other)
    out["gender_female"] = df["gender"].str.strip().str.lower().apply(
        lambda g: 1 if g == "female" else 0
    )

    # 5. One-hot: irrigation_type 
    # Valid values: "none", "canal", "drip"
    irr = df["irrigation_type"].str.strip().str.lower()
    out["irr_none"]  = (irr == "none").astype(int)
    out["irr_canal"] = (irr == "canal").astype(int)
    out["irr_drip"]  = (irr == "drip").astype(int)

    # 6. One-hot: insurance_type ─
    # Valid values: "none", "crop", "livestock", "both"
    ins = df["insurance_type"].str.strip().str.lower()
    out["ins_none"]      = (ins == "none").astype(int)
    out["ins_crop"]      = (ins == "crop").astype(int)
    out["ins_livestock"] = (ins == "livestock").astype(int)
    out["ins_both"]      = (ins == "both").astype(int)

    # 7. Multi-hot: crop_types 
    # Raw column is comma-separated, e.g. "staple,cash_crop"
    # A farmer can grow multiple crop types simultaneously → multi-hot encoding
    def has_crop(crop_str, crop_type: str) -> int:
        types = [c.strip().lower() for c in str(crop_str).split(",")]
        return int(crop_type in types)

    out["has_staple"]    = df["crop_types"].apply(has_crop, crop_type="staple")
    out["has_cash_crop"] = df["crop_types"].apply(has_crop, crop_type="cash_crop")
    out["has_vegetable"] = df["crop_types"].apply(has_crop, crop_type="vegetable")
    out["has_other"]     = df["crop_types"].apply(has_crop, crop_type="other")

    # 8. Labels (training data only) 
    if not inference:
        if "credit_score" in df.columns:
            out["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce").clip(0, 100)
        else:
            print("  [WARNING] 'credit_score' column not found — omitting from output.")

        if "creditworthiness" in df.columns:
            out["creditworthiness_code"] = df["creditworthiness"].str.strip().map(CREDITWORTHINESS_CODE)
            unmapped = out["creditworthiness_code"].isna().sum()
            if unmapped > 0:
                print(f"  [WARNING] {unmapped} rows had unrecognised creditworthiness values → NaN")
        else:
            print("  [WARNING] 'creditworthiness' column not found — omitting from output.")

    # 9. Enforce stable output ordering 
    ordered = [c for c in OUTPUT_COLS if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    out = out[ordered + extras]

    # 10. Missing value report 
    null_counts = out.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        print("\n  [WARNING] Missing values detected after preprocessing:")
        for col, n in null_cols.items():
            print(f"    {col:35s}: {n} null(s)  ({n/len(out)*100:.1f}%)")
        print("  Consider imputation before model training.\n")

    return out


#  I/O HELPERS

def load_input(path: str, sheet: str | None = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        sheet_name = sheet or "🌾 Farmer Data"
        xl = pd.ExcelFile(path)
        # Find sheet by exact name or partial match
        if sheet_name in xl.sheet_names:
            target_sheet = sheet_name
        else:
            matches = [s for s in xl.sheet_names if sheet_name.lower().replace("🌾 ","") in s.lower()]
            if not matches:
                raise ValueError(
                    f"Sheet '{sheet_name}' not found. Available sheets: {xl.sheet_names}"
                )
            target_sheet = matches[0]
            print(f"  [INFO] Loaded sheet: '{target_sheet}'")
        # The GrowForMe workbook has 2 title/group rows above the real header (row index 2).
        # Auto-detect: scan first 5 rows for one that contains 'farmer_id'
        probe = pd.read_excel(path, sheet_name=target_sheet, header=None, nrows=6)
        header_row = 0
        for i, row in probe.iterrows():
            # Match on normalised form: "Farmer Id" -> "farmer_id", "farmer_id" -> "farmer_id"
            normed = [str(v).strip().lower().replace(" ", "_") for v in row.values]
            if "farmer_id" in normed:
                header_row = i
                break
        df = pd.read_excel(path, sheet_name=target_sheet, header=header_row)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .xlsx")

    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"  [INFO] Loaded {len(df):,} rows × {len(df.columns)} columns from '{path}'")
    return df


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return base + "_ml_ready.csv"


def validate_against_metadata_schema(df: pd.DataFrame, metadata_path: str) -> None:
    if not os.path.exists(metadata_path):
        print(f"  [INFO] Metadata not found at '{metadata_path}' - skipping schema validation.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected = metadata.get("feature_columns", [])
    got = [c for c in df.columns if c not in {"farmer_id", "credit_score", "creditworthiness_code"}]

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))

    if missing or extra:
        raise ValueError(
            "Preprocessing output does not match trained model schema.\n"
            f"Missing features: {missing}\n"
            f"Unexpected features: {extra}"
        )

    print(f"  [INFO] Schema validation passed against '{metadata_path}'.")


#  MAIN

def main():
    parser = argparse.ArgumentParser(
        description="GrowForMe — Farmer Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input",     required=True,  help="Path to raw farmer data (.csv or .xlsx)")
    parser.add_argument("--output",    default=None,    help="Path for ML-ready output CSV (default: <input>_ml_ready.csv)")
    parser.add_argument("--sheet",     default=None,    help="Excel sheet name (default: '🌾 Farmer Data')")
    parser.add_argument("--inference", action="store_true",
                        help="Inference mode: skip credit_score / creditworthiness label columns")
    parser.add_argument("--preview",   action="store_true",
                        help="Preview first 5 rows and column list without saving")
    parser.add_argument("--metadata",  default="artifacts/model_metadata.json",
                        help="Optional metadata JSON to validate output schema against trained model")
    args = parser.parse_args()

    print("\n═══════════════════════════════════════════════════════════════")
    print("  GrowForMe Credit Scoring — Preprocessing Pipeline")
    print("═══════════════════════════════════════════════════════════════\n")

    # Load
    raw_df = load_input(args.input, args.sheet)

    # Transform
    print("\n  [INFO] Running feature engineering...")
    ml_df = preprocess(raw_df, inference=args.inference)

    # Summary
    n_features = len(ml_df.columns) - 1  # exclude farmer_id
    if not args.inference and "credit_score" in ml_df.columns:
        n_features -= 2  # exclude labels
    print(f"\n  [INFO] Output shape  : {ml_df.shape[0]:,} rows × {ml_df.shape[1]} columns")
    print(f"  [INFO] Feature count : {n_features} (excl. farmer_id and labels)")
    print(f"  [INFO] Mode          : {'inference (no labels)' if args.inference else 'training (labels included)'}")
    validate_against_metadata_schema(ml_df, args.metadata)

    if args.preview:
        print("\n─Column list ──")
        for i, col in enumerate(ml_df.columns, 1):
            print(f"  {i:2d}. {col}")
        print("\n─First 5 rows ──")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(ml_df.head().to_string(index=False))
        print("\n  [INFO] Preview mode — file not saved. Remove --preview to save.\n")
        return

    # Save
    out_path = args.output or default_output_path(args.input)
    ml_df.to_csv(out_path, index=False)
    print(f"\n  [INFO] Saved ML-ready CSV → {out_path}")
    print("\n  Next steps:")
    print("    X = df.drop(columns=['farmer_id', 'credit_score', 'creditworthiness_code'])")
    print("    y = df['credit_score']")
    print("    model.fit(X, y)\n")
    print("═══════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()