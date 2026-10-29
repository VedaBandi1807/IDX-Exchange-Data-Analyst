"""Week 6: feature engineering, market metrics, school districts and segment analysis.

Input:  Week 5 files (week5_outputs/<dataset>/<dataset>_cleaned_week5.csv).
Steps:
  1. Read the file in chunks (memory-safe).
  2. Engineer the metrics from the handbook:
       price_ratio                  = ClosePrice / ListPrice
       close_to_original_list_ratio = ClosePrice / OriginalListPrice
       price_per_sqft               = ClosePrice / LivingArea
       days_on_market               = DaysOnMarket (raw field)
       close_year, close_month, yrmo  from CloseDate
       listing_to_contract_days     = PurchaseContractDate - ListingContractDate
       contract_to_close_days       = CloseDate - PurchaseContractDate
     NOTE: the handbook gives ClosePrice / OriginalListPrice for BOTH "Price
     Ratio" and "Close to Original List Ratio". Here price_ratio uses ListPrice
     (the final asking price) so the two metrics differ. Change PRICE_RATIO_BASE
     to "OriginalListPrice" if your mentor wants the handbook formula exactly.
     Ratios/PPSF are only calculated when both values are > 0; otherwise blank.
     Day counts are kept as-is (negative values are already flagged in Week 5).
  3. (Optional) Add school districts from latitude/longitude with a spatial join.
  4. Save the enriched dataset and a sample output table.
  5. Segment analysis (sold data): summary statistics grouped by
     PropertyType/PropertySubType, CountyOrParish/MLSAreaMajor,
     ListOfficeName and BuyerOfficeName.
Run the whole script.
"""

# %% Step 0 - Setup
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")
INPUT_FOLDER = DATA_FOLDER / "week5_outputs"
OUTPUT_FOLDER = DATA_FOLDER / "week6_outputs"
DATASETS = ["sold", "listings"]
CHUNK_SIZE = 50_000
PRICE_RATIO_BASE = "ListPrice"  # or "OriginalListPrice" (handbook formula)

# School districts (optional). Download "California School District Areas
# 2024-25" from data.ca.gov as GeoJSON or Shapefile and put the path here.
# If the file is missing or geopandas is not installed, this step is skipped.
SCHOOL_DISTRICT_FILE = DATA_FOLDER / "DistrictAreas2425" / "DistrictAreas2425.shp"
SCHOOL_NAME_FIELD = "DistrictNa"  # shapefile shortens DistrictName to DistrictNa
SCHOOL_TYPE_FIELD = "DistrictTy"  # shapefile shortens DistrictType to DistrictTy
# DistrictType value in the file -> output column name
SCHOOL_DISTRICT_COLUMNS = {
    "Elementary": "elementary_district",
    "High": "high_district",
    "Unified": "unified_district",
}

NEW_COLUMNS = ["price_ratio", "close_to_original_list_ratio", "price_per_sqft",
               "days_on_market", "close_year", "close_month", "yrmo",
               "listing_to_contract_days", "contract_to_close_days"]
SEGMENTS = {
    "property_type": ["PropertyType", "PropertySubType"],
    "county_area": ["CountyOrParish", "MLSAreaMajor"],
    "list_office": ["ListOfficeName"],
    "buyer_office": ["BuyerOfficeName"],
}


def number(chunk, column):
    if column not in chunk:
        return pd.Series(float("nan"), index=chunk.index)
    return pd.to_numeric(chunk[column], errors="coerce")


def date(chunk, column):
    if column not in chunk:
        return pd.Series(pd.NaT, index=chunk.index)
    return pd.to_datetime(chunk[column], errors="coerce")


def safe_ratio(top, bottom):
    # Only divide when both values are positive; otherwise leave blank.
    return (top / bottom).where((top > 0) & (bottom > 0))


# %% Step 2 - Engineered metrics
def add_metrics(chunk):
    close_price = number(chunk, "ClosePrice")
    close_date = date(chunk, "CloseDate")
    listed = date(chunk, "ListingContractDate")
    contract = date(chunk, "PurchaseContractDate")

    chunk["price_ratio"] = safe_ratio(close_price, number(chunk, PRICE_RATIO_BASE))
    chunk["close_to_original_list_ratio"] = safe_ratio(close_price,
                                                       number(chunk, "OriginalListPrice"))
    chunk["price_per_sqft"] = safe_ratio(close_price, number(chunk, "LivingArea"))
    chunk["days_on_market"] = number(chunk, "DaysOnMarket")
    chunk["close_year"] = close_date.dt.year.astype("Int64")
    chunk["close_month"] = close_date.dt.month.astype("Int64")
    chunk["yrmo"] = close_date.dt.strftime("%Y-%m")
    chunk["listing_to_contract_days"] = (contract - listed).dt.days.astype("Int64")
    chunk["contract_to_close_days"] = (close_date - contract).dt.days.astype("Int64")


# %% Step 3 - School districts (optional spatial join)
def load_school_districts():
    if not SCHOOL_DISTRICT_FILE.is_file():
        print(f"School districts skipped: file not found ({SCHOOL_DISTRICT_FILE}).")
        return None
    try:
        import geopandas as gpd
    except ImportError:
        print("School districts skipped: run 'pip install geopandas' first.")
        return None
    districts = gpd.read_file(SCHOOL_DISTRICT_FILE).to_crs("EPSG:4326")
    print(f"School district columns: {list(districts.columns)}")
    name_field = SCHOOL_NAME_FIELD or next(
        (c for c in districts.columns if "name" in c.lower()), None)
    type_field = SCHOOL_TYPE_FIELD or next(
        (c for c in districts.columns if "type" in c.lower()), None)
    if name_field is None or type_field is None:
        print("School districts skipped: set SCHOOL_NAME_FIELD and SCHOOL_TYPE_FIELD.")
        return None
    print(f"Using name field '{name_field}' and type field '{type_field}'.")
    districts = districts[[name_field, type_field, "geometry"]].rename(
        columns={name_field: "district_name", type_field: "district_type"})
    print(f"District types in file: {districts['district_type'].value_counts().to_dict()}")
    return districts


def add_school_districts(chunk, districts):
    import geopandas as gpd
    lat, lon = number(chunk, "Latitude"), number(chunk, "Longitude")
    # Only use coordinates that passed the Week 5 geographic checks.
    bad = chunk.get("any_geo_issue_flag", pd.Series("False", index=chunk.index))
    usable = lat.notna() & lon.notna() & bad.astype(str).str.lower().ne("true")
    points = gpd.GeoDataFrame(index=chunk.index[usable],
                              geometry=gpd.points_from_xy(lon[usable], lat[usable]),
                              crs="EPSG:4326")
    joined = gpd.sjoin(points, districts, how="left", predicate="within")
    # A home can sit in both an elementary and a high school district, so the
    # join can return several rows per home. Put each district type in its own
    # column so every name stays matched to its type.
    joined = joined.dropna(subset=["district_name"])
    for district_type, column in SCHOOL_DISTRICT_COLUMNS.items():
        names = joined.loc[joined["district_type"].astype(str).str.strip() == district_type,
                           "district_name"]
        chunk[column] = names.groupby(level=0).agg(
            lambda s: "; ".join(sorted(s.astype(str).unique())))


# %% Step 5 - Segment analysis (sold data)
def segment_analysis(path, folder):
    needed = sorted({c for cols in SEGMENTS.values() for c in cols})
    header = pd.read_csv(path, nrows=0).columns
    metric_cols = ["ClosePrice", "price_per_sqft", "days_on_market", "price_ratio",
                   "close_to_original_list_ratio", "listing_to_contract_days",
                   "contract_to_close_days"]
    usecols = [c for c in needed + metric_cols if c in header]
    df = pd.concat(pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE,
                               dtype={c: str for c in needed if c in header}),
                   ignore_index=True)
    df = df[pd.to_numeric(df["ClosePrice"], errors="coerce") > 0]  # valid sales only
    for segment, group_cols in SEGMENTS.items():
        group_cols = [c for c in group_cols if c in df]
        if not group_cols:
            continue
        table = (df.groupby(group_cols, dropna=False)
                 .agg(sales=("ClosePrice", "size"),
                      median_close_price=("ClosePrice", "median"),
                      mean_close_price=("ClosePrice", "mean"),
                      median_price_per_sqft=("price_per_sqft", "median"),
                      median_days_on_market=("days_on_market", "median"),
                      median_price_ratio=("price_ratio", "median"),
                      median_close_to_original_list_ratio=("close_to_original_list_ratio", "median"),
                      median_listing_to_contract_days=("listing_to_contract_days", "median"),
                      median_contract_to_close_days=("contract_to_close_days", "median"))
                 .sort_values("sales", ascending=False)
                 .round(3))
        table.to_csv(folder / f"segment_{segment}.csv")
        print(f"\nSegment: {' + '.join(group_cols)} (top 10 by sales)")
        print(table.head(10).to_string())


# %% Steps 1-4 - Run one dataset
def engineer(name, districts):
    in_path = INPUT_FOLDER / name / f"{name}_cleaned_week5.csv"
    folder = OUTPUT_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{name}_features_week6.csv"
    print(f"\n===== {name} =====")
    rows, filled = 0, None
    for i, chunk in enumerate(pd.read_csv(in_path, dtype=str, encoding="utf-8",
                                          chunksize=CHUNK_SIZE)):
        add_metrics(chunk)
        if districts is not None:
            add_school_districts(chunk, districts)
        new_cols = [c for c in NEW_COLUMNS + list(SCHOOL_DISTRICT_COLUMNS.values())
                    if c in chunk]
        counts = chunk[new_cols].notna().sum()
        filled = counts if filled is None else filled + counts
        rows += len(chunk)
        if i == 0:  # Step 4 - sample output table showing the new columns
            sample_cols = [c for c in ["ListingKey", "ClosePrice", "ListPrice",
                                       "OriginalListPrice", "LivingArea", "CloseDate",
                                       "ListingContractDate", "PurchaseContractDate"]
                           if c in chunk] + new_cols
            sample = chunk.loc[number(chunk, "ClosePrice") > 0, sample_cols].head(10)
            sample.to_csv(folder / "sample_output.csv", index=False)
            print("Sample of new columns:")
            print(sample.to_string(index=False))
        chunk.to_csv(out_path, mode="w" if i == 0 else "a", header=(i == 0),
                     index=False, encoding="utf-8")

    populated = pd.DataFrame({"rows_populated": filled.astype(int),
                              "percent_populated": (filled / rows * 100).round(2)})
    populated.index.name = "new_column"
    populated.to_csv(folder / "new_column_population.csv")
    print(f"\nRows: {rows:,}. How many rows each new column was filled for:")
    print(populated.to_string())
    print(f"Saved: {out_path}")
    if name == "sold":
        segment_analysis(out_path, folder)


if __name__ == "__main__":
    school_districts = load_school_districts()
    for dataset in DATASETS:
        engineer(dataset, school_districts)
    print("\nWeek 6 feature engineering complete.")