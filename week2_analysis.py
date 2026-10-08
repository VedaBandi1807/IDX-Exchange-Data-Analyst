"""Week 2: use combined Residential files for analysis.

Only original property-type counts/shares come from monthly source files.
Run the whole script. Outputs replace earlier reports in week2_outputs.
"""

# %% Dataset Understanding
# Inspect combined datasets, field types, field roles, and repeated columns.
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")
OUTPUT_FOLDER = DATA_FOLDER / "week2_outputs"
INPUT_FILES = {
    "sold": "CRMLSSold_Combined_Residential_1.csv",
    "listings": "CRMLSListing_Combined_Residential_1.csv",
}
# Must match the months used to create the combined files.
START_MONTH = "2024-01"
END_MONTH = "2026-08"
NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]
DATE_FIELDS = ["CloseDate", "ListingContractDate", "PurchaseContractDate",
               "ContractStatusChangeDate"]
IDENTIFIERS = {"ListingKey", "ListingKeyNumeric", "ListingId"}
MARKET_FIELDS = set(NUMERIC_FIELDS + DATE_FIELDS) | {
    "PropertyType", "PropertySubType", "MlsStatus", "Latitude", "Longitude",
    "City", "CountyOrParish", "PostalCode", "StateOrProvince", "MLSAreaMajor",
    "ListAgentFullName", "ListAgentFirstName", "ListAgentLastName",
    "ListOfficeName", "BuyerOfficeName",
}


def read_csv_safely(path, usecols=None):
    # Strings preserve identifiers and leading zeros. Analysis conversions
    # are performed separately and do not overwrite source columns.
    try:
        df = pd.read_csv(path, dtype="string", encoding="utf-8", usecols=usecols)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype="string", encoding="cp1252", usecols=usecols)
    return df.replace(r"^\s*$", pd.NA, regex=True)


def base_column(column):
    base, separator, suffix = column.rpartition(".")
    return base if separator and suffix.isdigit() else column


def column_role(column):
    base = base_column(column)
    if base in IDENTIFIERS:
        return "Identifier / metadata"
    if base in MARKET_FIELDS:
        return "Market analysis / validation"
    if base.startswith("OriginatingSystem") or "Email" in base:
        return "Source / contact metadata"
    if base in {"latfilled", "lonfilled"}:
        return "Supplemental coordinates: confirm definition"
    return "Other property / agent attribute: review definition"


def dataset_understanding(df, folder):
    review, duplicates = [], []
    for column in df.columns:
        base = base_column(column)
        expected = "numeric" if base in NUMERIC_FIELDS else (
            "datetime" if base in DATE_FIELDS else "text / identifier / category"
        )
        review.append({"column": column, "loaded_dtype": str(df[column].dtype),
                       "expected_type": expected, "field_role": column_role(column)})
        if base != column and base in df.columns:
            left, right = df[base].str.strip(), df[column].str.strip()
            equal = left.eq(right).fillna(False)
            duplicates.append({
                "original": base, "repeated": column,
                "conflicting_values": int((left.notna() & right.notna() & ~equal).sum()),
                "original_only": int((left.notna() & right.isna()).sum()),
                "repeated_only": int((left.isna() & right.notna()).sum()),
                "all_match": bool((equal | (left.isna() & right.isna())).all()),
            })
    pd.DataFrame(review).to_csv(folder / "column_review.csv", index=False)
    pd.DataFrame(duplicates, columns=[
        "original", "repeated", "conflicting_values", "original_only",
        "repeated_only", "all_match",
    ]).to_csv(folder / "duplicate_column_report.csv", index=False)
    df.head().to_csv(folder / "first_five_rows.csv", index=False)
    df["PropertyType"].value_counts(dropna=False).rename_axis("PropertyType").reset_index(
        name="rows"
    ).to_csv(folder / "combined_property_type_counts.csv", index=False)


# %% Missing Value Analysis
# Count nulls, flag >90% missing, and document retention recommendations.
def missing_value_analysis(df, folder):
    report = pd.DataFrame({"missing_count": df.isna().sum(),
                           "missing_percent": df.isna().mean() * 100})
    report.index.name = "column"
    report["above_90_percent_missing"] = report["missing_percent"] > 90
    decisions = []
    for column in report.index:
        if base_column(column) != column:
            decision = "Review duplicate comparison before merging or dropping"
        elif column in MARKET_FIELDS | IDENTIFIERS:
            decision = "Retain: core analysis, validation, or identification field"
        elif column in {"latfilled", "lonfilled"}:
            decision = "Retain pending confirmation of coordinate definitions"
        elif report.loc[column, "above_90_percent_missing"]:
            decision = "Candidate to drop from analysis copy after relevance review"
        else:
            decision = "Retain pending field-definition review"
        decisions.append(decision)
    report["retention_recommendation"] = decisions
    report = report.sort_values("missing_percent", ascending=False)
    report.to_csv(folder / "missing_value_report.csv")
    high = report.loc[report["above_90_percent_missing"]]
    high.to_csv(folder / "high_missing_columns.csv")
    return len(high)


# %% Numeric Distribution Review
# Summaries and full-range plots for all nine fields; no outliers are removed.
def numeric_distribution_review(df, folder):
    numeric, summaries = {}, []
    charts = folder / "charts"
    charts.mkdir(exist_ok=True)
    for column in NUMERIC_FIELDS:
        if column not in df:
            summaries.append({"column": column, "status": "Column absent"})
            continue
        parsed = pd.to_numeric(df[column], errors="coerce")
        infinite = parsed.isin([float("inf"), float("-inf")])
        numeric[column] = parsed.mask(infinite)
        values = numeric[column].dropna().astype(float)
        summary = {
            "column": column, "valid_numeric_count": len(values),
            "missing_source_count": int(df[column].isna().sum()),
            "invalid_numeric_count": int((df[column].notna() & parsed.isna()).sum()),
            "infinite_count": int(infinite.sum()),
        }
        if not values.empty:
            q1, q3 = values.quantile(.25), values.quantile(.75)
            lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            summary.update({
                "min": values.min(), "max": values.max(), "mean": values.mean(),
                "median": values.median(), "std": values.std(),
                "iqr_lower_bound": lower, "iqr_upper_bound": upper,
                "potential_outlier_count": int(((values < lower) | (values > upper)).sum()),
            })
            for percentile in [1, 5, 25, 50, 75, 95, 99]:
                summary[f"p{percentile:02}"] = values.quantile(percentile / 100)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(values, bins=50, edgecolor="white")
            axes[0].set(xlabel=column, ylabel="Records", title="Full-range histogram")
            axes[1].boxplot(values)
            axes[1].set(ylabel=column, title="Full-range boxplot", xticks=[])
            fig.suptitle(f"{folder.name}: {column}")
            fig.tight_layout()
            fig.savefig(charts / f"{column}.png", dpi=150)
            plt.close(fig)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(folder / "numeric_summary.csv", index=False)
    return numeric


# %% Suggested Intern Questions
# Only question 1 reads monthly files; all other questions use combined data.

# 1. What is the original Residential vs. other property-type share?
def original_property_type_shares(name, folder, combined_residential_rows):
    prefix = "CRMLSSold" if name == "sold" else "CRMLSListing"
    counts = Counter()
    total, missing_types = 0, 0
    monthly, missing_files = [], []
    for month in pd.period_range(START_MONTH, END_MONTH, freq="M").strftime("%Y%m"):
        normal = DATA_FOLDER / f"{prefix}{month}.csv"
        filled = DATA_FOLDER / f"{prefix}{month}_filled.csv"
        path = normal if normal.is_file() else filled
        if not path.is_file():
            missing_files.append(f"{normal.name} OR {filled.name}")
            continue
        # Read this one column only. Never append monthly records to combined data.
        types = read_csv_safely(path, usecols=["PropertyType"])["PropertyType"].str.strip()
        counts.update(types.dropna().value_counts().to_dict())
        total += len(types)
        missing_types += int(types.isna().sum())
        monthly.append({"file": path.name, "rows": len(types),
                        "Residential_rows": int(types.eq("Residential").fillna(False).sum())})
    pd.DataFrame(monthly, columns=["file", "rows", "Residential_rows"]).to_csv(
        folder / "property_type_monthly_inputs.csv", index=False)
    complete = not missing_files
    details = [{"PropertyType": key, "rows": value} for key, value in sorted(counts.items())]
    details.append({"PropertyType": pd.NA, "rows": missing_types})
    detail = pd.DataFrame(details)
    detail["percent_of_available_rows"] = detail["rows"] / total * 100 if total else float("nan")
    detail["all_required_months_present"] = complete
    detail.to_csv(folder / "original_property_type_share.csv", index=False)
    residential = counts.get("Residential", 0)
    shares = pd.DataFrame({
        "category": ["Residential", "Other known types", "Missing type"],
        "rows": [residential, total - residential - missing_types, missing_types],
    })
    shares["percent_of_available_rows"] = shares["rows"] / total * 100 if total else float("nan")
    shares["all_required_months_present"] = complete
    shares.to_csv(folder / "residential_vs_other_share.csv", index=False)
    notes = [f"Property-type shares use monthly files from {START_MONTH} through {END_MONTH}.",
             f"Monthly files read: {len(monthly)}; missing: {len(missing_files)}.",
             f"Monthly Residential rows: {residential:,}; combined Residential rows: {combined_residential_rows:,}.",
             "Normal monthly files take precedence over _filled files; each month is counted once."]
    if missing_files:
        notes.append("WARNING: shares cover available months only; they do not represent the full requested period.")
        notes.extend(missing_files)
    elif residential != combined_residential_rows:
        notes.append("WARNING: counts differ. Check month coverage and source file versions; combined data was not changed.")
    else:
        notes.append("Residential counts agree; count agreement alone does not prove identical records or month coverage.")
    (folder / "property_type_coverage.txt").write_text("\n".join(notes), encoding="utf-8")
    return notes


def suggested_intern_questions(df, numeric, name, folder):
    notes = original_property_type_shares(name, folder, len(df))
    # 2. Median and average close prices (sold dataset).
    # 3. DaysOnMarket distribution is reported by the numeric review above.
    notes.append("DaysOnMarket: see numeric_summary.csv and charts/DaysOnMarket.png.")
    if name == "sold" and "ClosePrice" in numeric:
        close = numeric["ClosePrice"]
        notes.extend([f"Mean close price (finite numeric values): {close.mean():,.2f}",
                      f"Median close price (finite numeric values): {close.median():,.2f}"])
        # 4. Percentage sold above, at, or below ListPrice.
        if "ListPrice" in numeric:
            listed = numeric["ListPrice"]
            valid = (close.gt(0) & listed.gt(0)).fillna(False)
            denominator = int(valid.sum())
            rows = []
            for label, mask in [("Above", close > listed), ("At", close == listed), ("Below", close < listed)]:
                count = int((valid & mask.fillna(False)).sum())
                rows.append({"relative_to_ListPrice": label, "rows": count,
                             "percent": count / denominator * 100 if denominator else None,
                             "eligible_sales": denominator})
            pd.DataFrame(rows).to_csv(folder / "sold_vs_list_price.csv", index=False)
            notes.append(f"Sold-vs-list denominator: {denominator:,} rows with positive finite prices; "
                         f"{len(df) - denominator:,} excluded from this calculation only. Uses ListPrice.")
        # 6. Counties with the highest median close prices; include sample sizes.
        if "CountyOrParish" in df:
            county = pd.DataFrame({"county": df["CountyOrParish"].str.strip(), "ClosePrice": close})
            county = county.loc[close.gt(0).fillna(False)]
            county.groupby("county")["ClosePrice"].agg(
                sales_count="count", median_close_price="median", mean_close_price="mean"
            ).sort_values("median_close_price", ascending=False).to_csv(folder / "county_price_ranking.csv")
            notes.append("County rankings use positive finite close prices and known counties; review sales counts.")
    # 5. Date consistency and parsing problems (both datasets).
    dates, checks = {}, []
    for column in DATE_FIELDS:
        if column in df:
            dates[column] = pd.to_datetime(df[column], errors="coerce", utc=True, format="mixed")
            failures = int((df[column].notna() & dates[column].isna()).sum())
            notes.append(f"{column}: {failures:,} nonblank values failed date parsing.")
    for earlier, later in [("ListingContractDate", "CloseDate"),
                           ("ListingContractDate", "PurchaseContractDate"),
                           ("PurchaseContractDate", "CloseDate")]:
        if earlier in dates and later in dates:
            valid = dates[earlier].notna() & dates[later].notna()
            checks.append({"check": f"{earlier} after {later}",
                           "comparable_rows": int(valid.sum()),
                           "violations": int((valid & (dates[earlier] > dates[later])).sum())})
    pd.DataFrame(checks, columns=["check", "comparable_rows", "violations"]).to_csv(
        folder / "date_consistency.csv", index=False)
    (folder / "eda_findings.txt").write_text("\n".join(notes), encoding="utf-8")
    return notes


# Run the four sections and save the Week 2 deliverables.
def analyze(name, filename):
    folder = OUTPUT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    df = read_csv_safely(DATA_FOLDER / filename)
    if "PropertyType" not in df:
        raise ValueError(f"{filename} is missing PropertyType.")
    before = len(df)
    dataset_understanding(df, folder)
    # Repeat the documented filter as validation of the Week 1 inputs.
    mask = df["PropertyType"].str.strip().eq("Residential").fillna(False)
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"{filename} contains no Residential rows.")
    high_missing = missing_value_analysis(df, folder)
    numeric = numeric_distribution_review(df, folder)
    notes = suggested_intern_questions(df, numeric, name, folder)
    df.to_csv(folder / f"{name}_Residential_week2.csv", index=False, encoding="utf-8")
    overview = [f"Dataset: {name}", f"Main input: {filename}",
                f"Rows before Residential filter: {before:,}",
                f"Rows after Residential filter: {len(df):,}", f"Columns: {len(df.columns):,}",
                f"Columns above 90% missing: {high_missing}",
                "Main analysis uses combined data only; monthly files supply original property-type shares only.",
                "Loaded types are strings; numeric/date validity is checked separately for analysis.",
                "No columns or outliers removed. Retention recommendations need review.",
                "Missing sale fields may be expected for unsold listings."]
    (folder / "dataset_overview.txt").write_text("\n".join(overview), encoding="utf-8")
    print("\n".join(overview + notes))
    print(f"Results saved to: {folder}\n")


if __name__ == "__main__":
    for dataset_name, combined_filename in INPUT_FILES.items():
        analyze(dataset_name, combined_filename)
    print("Week 2 analysis complete.")
