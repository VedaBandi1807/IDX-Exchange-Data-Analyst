"""Week 5: data cleaning, part 2 - date consistency and geographic checks.

Input:  Week 4 cleaned files (week4_outputs/<dataset>/<dataset>_cleaned_week4.csv).
Steps:
  1. Read the file in chunks (memory-safe, same approach as Week 4).
  2. Date consistency flags:
       listing_after_close_flag  - ListingContractDate is after CloseDate
       purchase_after_close_flag - PurchaseContractDate is after CloseDate
       negative_timeline_flag    - ListingContractDate is after PurchaseContractDate
                                   (offer accepted before the home was listed,
                                   i.e. negative listing-to-contract days)
  3. Geographic flags:
       missing_coords_flag       - Latitude or Longitude is blank
       zero_coords_flag          - Latitude = 0 or Longitude = 0 (placeholder values)
       positive_longitude_flag   - Longitude > 0 (California longitudes are negative)
       out_of_state_coords_flag  - coordinates outside a California bounding box
       any_geo_issue_flag        - any of the above
  4. Missing-value summary for core fields (no values are filled in or guessed;
     e.g. listings that have not sold are expected to have no ClosePrice).
  5. Save the dataset with the new flag columns plus summary reports.
Rows are flagged, NOT removed. Run the whole script.
"""

# %% Step 0 - Setup
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")
INPUT_FOLDER = DATA_FOLDER / "week4_outputs"
OUTPUT_FOLDER = DATA_FOLDER / "week5_outputs"
DATASETS = ["sold", "listings"]
CHUNK_SIZE = 50_000  # lower this if you get memory errors
DATE_FIELDS = ["CloseDate", "PurchaseContractDate", "ListingContractDate",
               "ContractStatusChangeDate"]
# Approximate California bounding box (degrees). Points outside are implausible.
CA_LAT = (32.5, 42.1)
CA_LON = (-124.5, -114.1)
CORE_FIELDS = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
               "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket",
               "YearBuilt", "CloseDate", "ListingContractDate",
               "PurchaseContractDate", "Latitude", "Longitude",
               "CountyOrParish", "City", "PostalCode", "rate_30yr_fixed"]
DATE_FLAGS = ["listing_after_close_flag", "purchase_after_close_flag",
              "negative_timeline_flag"]
GEO_FLAGS = ["missing_coords_flag", "zero_coords_flag", "positive_longitude_flag",
             "out_of_state_coords_flag", "any_geo_issue_flag"]


def read_chunks(path):
    # Week 4 saved the file as UTF-8. Read as text, then convert what we need.
    return pd.read_csv(path, dtype=str, encoding="utf-8", chunksize=CHUNK_SIZE)


def date_or_blank(chunk, column):
    # Missing column -> all blank, so the comparison is simply False.
    if column not in chunk:
        return pd.Series(pd.NaT, index=chunk.index)
    return pd.to_datetime(chunk[column], errors="coerce")


# %% Step 2 - Date consistency flags
def add_date_flags(chunk):
    listed = date_or_blank(chunk, "ListingContractDate")
    contract = date_or_blank(chunk, "PurchaseContractDate")
    closed = date_or_blank(chunk, "CloseDate")
    # Comparisons with a blank date are False, so blanks are never flagged.
    chunk["listing_after_close_flag"] = (listed > closed).fillna(False)
    chunk["purchase_after_close_flag"] = (contract > closed).fillna(False)
    chunk["negative_timeline_flag"] = (listed > contract).fillna(False)
    comparable = {
        "listing_after_close_flag": int((listed.notna() & closed.notna()).sum()),
        "purchase_after_close_flag": int((contract.notna() & closed.notna()).sum()),
        "negative_timeline_flag": int((listed.notna() & contract.notna()).sum()),
    }
    return comparable


# %% Step 3 - Geographic flags
def add_geo_flags(chunk):
    if "Latitude" not in chunk or "Longitude" not in chunk:
        raise ValueError("Latitude/Longitude columns are missing.")
    lat = pd.to_numeric(chunk["Latitude"], errors="coerce")
    lon = pd.to_numeric(chunk["Longitude"], errors="coerce")
    missing = lat.isna() | lon.isna()
    zero = (lat == 0) | (lon == 0)
    usable = ~missing & ~zero
    chunk["missing_coords_flag"] = missing
    chunk["zero_coords_flag"] = zero.fillna(False)
    chunk["positive_longitude_flag"] = (lon > 0).fillna(False)
    # Only checked for coordinates that are present and non-zero.
    outside = ~lat.between(*CA_LAT) | ~lon.between(*CA_LON)
    chunk["out_of_state_coords_flag"] = usable & outside
    chunk["any_geo_issue_flag"] = chunk[GEO_FLAGS[:-1]].any(axis=1)


# %% Steps 1-5 - Run one dataset
def validate(name):
    in_path = INPUT_FOLDER / name / f"{name}_cleaned_week4.csv"
    folder = OUTPUT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{name}_cleaned_week5.csv"
    print(f"\n===== {name} =====")

    rows, flag_totals, comparable_totals = 0, None, {f: 0 for f in DATE_FLAGS}
    missing, columns_in = None, None
    for i, chunk in enumerate(read_chunks(in_path)):
        if columns_in is None:
            columns_in = len(chunk.columns)
        rows += len(chunk)
        # Step 4 - missing values in core fields (counted before adding flags).
        present = [c for c in CORE_FIELDS if c in chunk]
        chunk_missing = chunk[present].isna().sum()
        missing = chunk_missing if missing is None else missing + chunk_missing

        for flag, n in add_date_flags(chunk).items():
            comparable_totals[flag] += n
        add_geo_flags(chunk)

        chunk_flags = chunk[DATE_FLAGS + GEO_FLAGS].sum()
        flag_totals = chunk_flags if flag_totals is None else flag_totals + chunk_flags
        # Step 5 - append each processed chunk to the output file.
        chunk.to_csv(out_path, mode="w" if i == 0 else "a", header=(i == 0),
                     index=False, encoding="utf-8")
        columns_out = len(chunk.columns)

    # Date flag summary
    date_summary = pd.DataFrame([
        {"flag": f, "flagged_rows": int(flag_totals[f]),
         "rows_with_both_dates": comparable_totals[f],
         "percent_of_comparable": round(flag_totals[f] / comparable_totals[f] * 100, 3)
         if comparable_totals[f] else None}
        for f in DATE_FLAGS])
    date_summary.to_csv(folder / "date_flag_counts.csv", index=False)

    # Geographic data quality summary
    geo_summary = pd.DataFrame([
        {"flag": f, "flagged_rows": int(flag_totals[f]),
         "percent_of_rows": round(flag_totals[f] / rows * 100, 3)} for f in GEO_FLAGS])
    geo_summary.to_csv(folder / "geo_quality_summary.csv", index=False)

    # Missing values in core fields
    missing_summary = pd.DataFrame({"missing_count": missing.astype(int),
                                    "missing_percent": (missing / rows * 100).round(2)})
    missing_summary.index.name = "column"
    missing_summary.to_csv(folder / "core_missing_summary.csv")

    pd.DataFrame([
        {"stage": "before Week 5", "rows": rows, "columns": columns_in},
        {"stage": "after Week 5", "rows": rows, "columns": columns_out},
    ]).to_csv(folder / "before_after_counts.csv", index=False)

    print(f"Rows: {rows:,} (unchanged - flagged, not removed)")
    print(f"Columns: {columns_in:,} -> {columns_out:,} (+{columns_out - columns_in} flag columns)")
    print("Date consistency flags:")
    print(date_summary.to_string(index=False))
    print("Geographic data quality:")
    print(geo_summary.to_string(index=False))
    print("Missing values in core fields:")
    print(missing_summary.to_string())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for dataset in DATASETS:
        validate(dataset)
    print("\nWeek 5 validation complete. Rows were flagged, not removed.")
