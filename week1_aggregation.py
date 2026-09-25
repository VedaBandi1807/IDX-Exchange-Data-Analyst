import pandas as pd
from pathlib import Path

DATA_FOLDER = Path(r"C:\Users\Veda_\Downloads\IDXExchange\csv")

months = pd.period_range(start="2024-01", end="2026-04", freq="M").strftime("%Y%m").tolist()

def load_and_combine(prefix):

    dataframes = []
    total_rows_before_concat = 0
    missing_files = []
    duplicate_files = []

    for month in months:

        normal_filename = f"{prefix}{month}.csv"
        filled_filename = f"{prefix}{month}_filled.csv"

        normal_filepath = DATA_FOLDER / normal_filename
        filled_filepath = DATA_FOLDER / filled_filename

        if normal_filepath.exists():

            filepath = normal_filepath
            filename = normal_filename

        elif filled_filepath.exists():

            filepath = filled_filepath
            filename = filled_filename

        else:

            missing_files.append(f"{normal_filename} OR {filled_filename}")
            print(f"MISSING: " f"{normal_filename} / {filled_filename}")
            continue

        df = pd.read_csv(filepath)

        print(f"{filename}: {len(df):,} rows")

        dataframes.append(df)
        total_rows_before_concat += len(df)

    if missing_files:

        for file in missing_files:
            print(f"  {file}")

    if not dataframes:
        raise FileNotFoundError(f"No files found for {prefix}.")

    print(f"Rows before concatenation: " f"{total_rows_before_concat:,}")

    combined = pd.concat(dataframes, ignore_index=True)

    print(f"Rows after concatenation: " f"{len(combined):,}")

    return combined

sold = load_and_combine("CRMLSSold")

listings = load_and_combine("CRMLSListing")

sold_before_filter = len(sold)

print(f"Rows before Residential filter for sold: " f"{sold_before_filter:,}")

sold = sold[sold["PropertyType"].astype(str).str.strip().eq("Residential")].copy()

sold_after_filter = len(sold)

print(f"Rows after Residential filter for sold: " f"{sold_after_filter:,}")

listings_before_filter = len(listings)

print(f"Rows before Residential filter for listing: " f"{listings_before_filter:,}")

listings = listings[listings["PropertyType"].astype(str).str.strip().eq("Residential")].copy()

listings_after_filter = len(listings)

print(f"Rows after Residential filter for listing: " f"{listings_after_filter:,}")

sold_output = DATA_FOLDER / "CRMLSSold_Combined_Residential.csv"

listing_output = DATA_FOLDER / "CRMLSListing_Combined_Residential.csv"

sold.to_csv(sold_output, index=False)

listings.to_csv(listing_output, index=False)