"""Week 4: data cleaning, part 1 (Option A).

Input:  the Week 3 enriched files (combined Residential data + mortgage rate).
Steps:
  1. Load data and record starting row/column counts.
  2. Drop redundant columns (exact duplicate ".1" columns) and
     high-missing (>90%) non-core columns. Every drop is logged with a reason.
  3. Convert the four date fields to datetime.
  4. Convert numeric fields to numbers.
  5. Flag invalid numeric values (rows are flagged, NOT removed).
  6. Save the cleaned dataset plus before/after and data-type reports.
Run the whole script. Outputs go to week4_outputs/<dataset>/.
"""

# %% Step 0 - Setup
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")
INPUT_FOLDER = DATA_FOLDER / "week3_outputs"
OUTPUT_FOLDER = DATA_FOLDER / "week4_outputs"
INPUT_FILES = {
    "sold": "CRMLSSold_Combined_Residential_1_with_rates.csv",
    "listings": "CRMLSListing_Combined_Residential_1_with_rates.csv",
}
DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate",
               "ContractStatusChangeDate"]
NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt", "Latitude", "Longitude", "rate_30yr_fixed",
]
# Core fields are kept even if mostly empty (handbook: "keep core fields
# even if partially missing"). Same list as Week 2, plus Week 3 columns.
CORE_FIELDS = set(NUMERIC_FIELDS + DATE_FIELDS) | {
    "ListingKey", "ListingKeyNumeric", "ListingId",
    "PropertyType", "PropertySubType", "MlsStatus",
    "City", "CountyOrParish", "PostalCode", "StateOrProvince", "MLSAreaMajor",
    "ListAgentFullName", "ListAgentFirstName", "ListAgentLastName",
    "ListOfficeName", "BuyerOfficeName",
    "latfilled", "lonfilled", "year_month",
}
MISSING_THRESHOLD = 90  # percent
# Rule name -> (column, test that marks a value as invalid)
INVALID_RULES = {
    "invalid_close_price_flag": ("ClosePrice", lambda s: s <= 0),
    "invalid_living_area_flag": ("LivingArea", lambda s: s <= 0),
    "invalid_days_on_market_flag": ("DaysOnMarket", lambda s: s < 0),
    "invalid_bedrooms_flag": ("BedroomsTotal", lambda s: s < 0),
    "invalid_bathrooms_flag": ("BathroomsTotalInteger", lambda s: s < 0),
}


def read_csv_safely(path):
    # Load everything as text first so nothing is silently changed on load.
    try:
        df = pd.read_csv(path, dtype="string", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype="string", encoding="cp1252")
    return df.replace(r"^\s*$", pd.NA, regex=True)


def base_column(column):
    # "CloseDate.1" -> "CloseDate"; other names are unchanged.
    base, separator, suffix = column.rpartition(".")
    return base if separator and suffix.isdigit() else column


# %% Step 2 - Drop redundant and high-missing columns
# Done BEFORE type conversion so duplicate columns are compared as raw text.
def drop_columns(df):
    log = []
    # 2a. Repeated columns (e.g. "City.1") that match the original on every row.
    for column in df.columns:
        base = base_column(column)
        if base == column or base not in df.columns:
            continue
        left, right = df[base].str.strip(), df[column].str.strip()
        same = left.eq(right).fillna(False) | (left.isna() & right.isna())
        if same.all():
            log.append({"column": column, "reason": f"Exact duplicate of {base}"})
        else:
            print(f"  KEPT {column}: differs from {base} on {int((~same).sum()):,} rows - review")
    # 2b. Non-core columns with more than 90% missing values.
    missing_pct = df.isna().mean() * 100
    already = {row["column"] for row in log}
    for column, pct in missing_pct.items():
        if column in already or base_column(column) in CORE_FIELDS:
            continue
        if pct > MISSING_THRESHOLD:
            log.append({"column": column, "reason": f"{pct:.1f}% missing, not a core field"})
    log = pd.DataFrame(log, columns=["column", "reason"])
    return df.drop(columns=log["column"]), log


# %% Step 3 - Convert date fields to datetime
def convert_dates(df):
    report = []
    for column in DATE_FIELDS:
        if column not in df:
            report.append({"column": column, "status": "absent"})
            continue
        raw = df[column]
        parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        df[column] = parsed.dt.tz_localize(None)
        report.append({"column": column, "status": "converted",
                       "blank": int(raw.isna().sum()),
                       "failed_to_parse": int((raw.notna() & parsed.isna()).sum()),
                       "min": df[column].min(), "max": df[column].max()})
    return report


# %% Step 4 - Convert numeric fields to numbers
def convert_numbers(df):
    report = []
    for column in NUMERIC_FIELDS:
        if column not in df:
            report.append({"column": column, "status": "absent"})
            continue
        raw = df[column]
        df[column] = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
        report.append({"column": column, "status": "converted",
                       "blank": int(raw.isna().sum()),
                       "failed_to_parse": int((raw.notna() & df[column].isna()).sum())})
    return report


# %% Step 5 - Flag invalid numeric values (flag only; nothing removed)
def flag_invalid(df):
    counts = []
    for flag, (column, is_invalid) in INVALID_RULES.items():
        if column not in df:
            continue
        df[flag] = is_invalid(df[column]).fillna(False).astype(bool)
        counts.append({"flag": flag, "column": column, "flagged_rows": int(df[flag].sum()),
                       "rows_with_value": int(df[column].notna().sum())})
    flags = [row["flag"] for row in counts]
    df["any_invalid_flag"] = df[flags].any(axis=1) if flags else False
    counts.append({"flag": "any_invalid_flag", "column": "(any of the above)",
                   "flagged_rows": int(df["any_invalid_flag"].sum()),
                   "rows_with_value": len(df)})
    return counts


# %% Steps 1 + 6 - Run everything for one dataset and save outputs
def clean(name, filename):
    folder = OUTPUT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name} =====")

    # Step 1 - load and record starting shape.
    df = read_csv_safely(INPUT_FOLDER / filename)
    rows_before, cols_before = df.shape
    print(f"Step 1  loaded: {rows_before:,} rows x {cols_before:,} columns")

    df, dropped = drop_columns(df)
    dropped.to_csv(folder / "dropped_columns.csv", index=False)
    print(f"Step 2  dropped {len(dropped):,} columns -> {df.shape[1]:,} remain")

    date_report = pd.DataFrame(convert_dates(df))
    date_report.to_csv(folder / "date_conversion.csv", index=False)
    print("Step 3  dates converted:")
    print(date_report.to_string(index=False))

    number_report = pd.DataFrame(convert_numbers(df))
    number_report.to_csv(folder / "numeric_conversion.csv", index=False)
    print("Step 4  numbers converted:")
    print(number_report.to_string(index=False))

    flag_counts = pd.DataFrame(flag_invalid(df))
    flag_counts.to_csv(folder / "invalid_value_flags.csv", index=False)
    print("Step 5  invalid value flags:")
    print(flag_counts.to_string(index=False))

    # Step 6 - save cleaned data and summary reports.
    rows_after, cols_after = df.shape
    assert rows_after == rows_before, "Rows changed - Week 4 should only flag, not remove."
    pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}).to_csv(
        folder / "dtype_confirmation.csv", index=False)
    summary = pd.DataFrame([
        {"stage": "before cleaning", "rows": rows_before, "columns": cols_before},
        {"stage": "after cleaning", "rows": rows_after, "columns": cols_after},
    ])
    summary.to_csv(folder / "before_after_counts.csv", index=False)
    out = folder / f"{name}_cleaned_week4.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Step 6  {rows_before:,} x {cols_before:,} -> {rows_after:,} x {cols_after:,} "
          f"(columns: {cols_after - cols_before:+,} incl. new flag columns)")
    print(f"Saved: {out}")


if __name__ == "__main__":
    for dataset_name, input_filename in INPUT_FILES.items():
        clean(dataset_name, input_filename)
    print("\nWeek 4 cleaning complete. Rows were flagged, not removed.")
