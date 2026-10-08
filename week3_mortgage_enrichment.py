from pathlib import Path
import pandas as pd
 
DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")
OUTPUT_FOLDER = DATA_FOLDER / "week3_outputs"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
# Same combined Residential files used in Week 2.
DATASETS = {
    "sold": ("CRMLSSold_Combined_Residential_1.csv", "CloseDate"),
    "listings": ("CRMLSListing_Combined_Residential_1.csv", "ListingContractDate"),
}
 
 
def read_csv_safely(path):
    # Strings preserve identifiers and leading zeros (same approach as Week 2).
    try:
        df = pd.read_csv(path, dtype="string", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype="string", encoding="cp1252")
    return df.replace(r"^\s*$", pd.NA, regex=True)
 
 
# %% Step 1 - Fetch MORTGAGE30US from FRED (CSV endpoint, no API key needed)
def fetch_mortgage_rates():
    cache = OUTPUT_FOLDER / "MORTGAGE30US_raw.csv"
    try:
        mortgage = pd.read_csv(FRED_URL)
        mortgage.to_csv(cache, index=False)  # keep a copy of what was fetched
        print(f"Fetched {len(mortgage):,} weekly rows from FRED.")
    except Exception as error:
        if not cache.is_file():
            raise RuntimeError(f"Could not fetch FRED data and no cached copy exists: {error}")
        print(f"WARNING: FRED fetch failed ({error}); using cached copy {cache.name}.")
        mortgage = pd.read_csv(cache)
    # FRED names the date column 'observation_date' (older files used 'DATE').
    mortgage.columns = ["date", "rate_30yr_fixed"]
    mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
    # FRED writes '.' for missing weeks; coerce those to NaN and drop them.
    mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["rate_30yr_fixed"], errors="coerce")
    mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])
    print(f"Weekly coverage: {mortgage['date'].min():%Y-%m-%d} to {mortgage['date'].max():%Y-%m-%d}")
    return mortgage
 
 
# %% Step 2 - Resample weekly rates to monthly averages
def to_monthly(mortgage):
    mortgage["year_month"] = mortgage["date"].dt.to_period("M")
    monthly = (mortgage.groupby("year_month")["rate_30yr_fixed"]
               .agg(rate_30yr_fixed="mean", weeks_in_month="count")
               .reset_index())
    monthly["rate_30yr_fixed"] = monthly["rate_30yr_fixed"].round(4)
    monthly.to_csv(OUTPUT_FOLDER / "mortgage_rate_monthly.csv", index=False)
    print(f"Monthly rate rows: {len(monthly):,}")
    return monthly[["year_month", "rate_30yr_fixed"]]
 
 
# %% Steps 3-5 - Build year_month key, merge, validate
def enrich(name, filename, date_column, monthly, report):
    df = read_csv_safely(DATA_FOLDER / filename)
    rows_before = len(df)
 
    # Step 3 - year_month key from the transaction date. Source column is unchanged.
    parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True, format="mixed")
    df["year_month"] = parsed.dt.tz_localize(None).dt.to_period("M")
    missing_date = int(df[date_column].isna().sum())
    unparseable = int((df[date_column].notna() & parsed.isna()).sum())
 
    # Step 4 - left merge; many_to_one guarantees no row duplication.
    enriched = df.merge(monthly, on="year_month", how="left", validate="many_to_one")
    assert len(enriched) == rows_before, "Row count changed during merge."
 
    # Step 5 - validation: every row should have a rate.
    null_rates = int(enriched["rate_30yr_fixed"].isna().sum())
    has_key = enriched["year_month"].notna()
    unmatched_months = sorted(enriched.loc[has_key & enriched["rate_30yr_fixed"].isna(),
                                           "year_month"].astype(str).unique())
    status = "PASS" if null_rates == 0 else "FAIL"
    print(f"\n[{name}] rows before merge: {rows_before:,} | after merge: {len(enriched):,}")
    print(f"[{name}] null rate_30yr_fixed after merge: {null_rates:,} -> {status}")
    if null_rates:
        print(f"  - blank {date_column}: {missing_date:,}")
        print(f"  - unparseable {date_column}: {unparseable:,}")
        print(f"  - months with no FRED rate: {unmatched_months or 'none'}")
 
    # Preview
    print(enriched[[date_column, "year_month", "ClosePrice" if name == "sold" else "ListPrice",
                    "rate_30yr_fixed"]].head())
 
    report.append({"dataset": name, "date_key": date_column, "rows_before": rows_before,
                   "rows_after": len(enriched), "blank_dates": missing_date,
                   "unparseable_dates": unparseable, "null_rates_after_merge": null_rates,
                   "months_without_rate": "; ".join(unmatched_months), "validation": status})
 
    enriched["year_month"] = enriched["year_month"].astype("string")  # e.g. 2024-01
    out = OUTPUT_FOLDER / filename.replace(".csv", "_with_rates.csv")
    enriched.to_csv(out, index=False, encoding="utf-8")
    print(f"[{name}] saved: {out}")
    return null_rates
 
 
if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    monthly_rates = to_monthly(fetch_mortgage_rates())
    report = []
    total_nulls = sum(enrich(name, file, date_col, monthly_rates, report)
                      for name, (file, date_col) in DATASETS.items())
    pd.DataFrame(report).to_csv(OUTPUT_FOLDER / "merge_validation.csv", index=False)
    if total_nulls:
        print("\nValidation FAILED: null rates remain. See merge_validation.csv; "
              "rows were kept, not dropped.")
    else:
        print("\nValidation PASSED: no null rate values in either enriched dataset.")
    print("Week 3 mortgage rate enrichment complete.")
 
