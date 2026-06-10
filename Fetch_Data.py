""" Kindly note, I could only find 1 minute data from the XAUUSD asset in the period stated, and thus I had to resample it to 4 Hrs.
    I made sure to avoid lookahead bias in the resampling by ensurng the 4 Hr candle only contained data from the lower timeframes that happened within those 4 hrs.
    The Data files were large and it was not possible to commit them to this repo.
    Here is the data Source; https://www.histdata.com/download-free-forex-historical-data/.
"""

import pandas as pd
import os
import sys

def create_multitimeframe_data(file_paths: list[str]) -> pd.DataFrame:
    """
    Loads and merges 1-minute data from multiple files, resamples to 5-minute
    base data, then creates M15, H1, and H4 features without lookahead bias.

    The logic ensures that for any 5-minute bar, the higher timeframe (HTF)
    data corresponds to the *previous completed* HTF candle.

    Args:
        file_paths (list[str]): A list of paths to the CSV data files.
                                The CSVs are expected to have columns like:
                                'Date', 'Time', 'O', 'H', 'L', 'C'.

    Returns:
        pd.DataFrame: A DataFrame with the 5-minute data and
                      additional columns for M15, H1, and H4 data from the
                      previous completed candle.
    """
    print("Loading and merging data files...")
    df_list = []
    for path in file_paths:
        # --- ADDED: Check if file exists before trying to read ---
        if not os.path.exists(path):
            print(f"WARNING: Data file not found at '{path}'. Skipping.", file=sys.stderr)
            continue

        try:
            # Use sep=None and engine='python' for automatic separator detection
            df_chunk = pd.read_csv(path, sep=None, engine='python')
            if df_chunk.empty:
                print(f"WARNING: Data file '{path}' is empty. Skipping.", file=sys.stderr)
                continue

            # --- NORMALIZE COLUMNS ---
            # Clean: strip whitespace, lowercase, remove MT4/5 brackets like <DATE>
            df_chunk.columns = df_chunk.columns.str.strip().str.lower().str.replace('[<>]', '', regex=True)
            
            # Standardize to Title Case for the rest of the script
            rename_map = {
                'date': 'Date', 'time': 'Time',
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'vol': 'Volume', 'volume': 'Volume', 'tickvol': 'Volume'
            }
            df_chunk.rename(columns=rename_map, inplace=True)
            
            df_list.append(df_chunk)
        except Exception as e:
            print(f"ERROR: Failed to read data file at '{path}': {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Check if any data was loaded and concatenate
    if not df_list:
        print("ERROR: No data files were loaded. Please check file paths and content.", file=sys.stderr)
        sys.exit(1)
    df = pd.concat(df_list, ignore_index=True)

    # 2. Create a proper DatetimeIndex and standardize columns
    # --- MODIFIED: More robust date parsing ---
    try:
        # Try a common MT4/5 format first for robustness.
        # This is often more reliable than letting pandas guess.
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    except ValueError:
        # If the specific format fails, fall back to pandas' general parser
        # which is slower but more flexible. 'coerce' will turn failures into NaT.
        print("INFO: Could not parse dates with specific format '%Y.%m.%d %H:%M'. "
              "Falling back to general date parser. This may be slower.", file=sys.stdout)
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    # --- ADDED: Check for and report parsing errors ---
    bad_date_mask = df['datetime'].isnull()
    if bad_date_mask.any():
        num_bad_dates = bad_date_mask.sum()
        print(f"\n--- WARNING: Found and removed {num_bad_dates} rows with unparseable dates. ---", file=sys.stderr)
        # To help debug, show an example of a row that failed
        print("  Example of a row that failed to parse:", file=sys.stderr)
        print(df[bad_date_mask].iloc[0][['Date', 'Time']], file=sys.stderr)
        print("  This is often caused by an unexpected date format in one of the source files.", file=sys.stderr)
        print("--------------------------------------------------------------------------------\n", file=sys.stderr)
        df = df[~bad_date_mask]

    df = df.set_index('datetime').sort_index()

    # Explicitly select the columns we need to ensure a clean DataFrame
    required_cols = ['Open', 'High', 'Low', 'Close']
    if 'Volume' in df.columns:
        required_cols.append('Volume')
    df = df[required_cols]

    print("Generating higher timeframe features...")
    # 3. Define higher timeframes and the aggregation logic
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'

    # Resample to 5-minute base timeframe
    print("  - Resampling to 5-minute base timeframe...")
    df = df.resample('5min', label='left', closed='left').agg(agg_dict).dropna()

    timeframes = {'15min': 'M15', '1h': 'H1', '4h': 'H4', '1D': 'D1'}

    # 4. Resample, shift, and merge for each timeframe
    for tf_pandas, tf_name in timeframes.items():
        print(f"  - Processing {tf_name} data...")
        # Resample data to the higher timeframe
        df_resampled = df.resample(tf_pandas, label='left', closed='left').agg(agg_dict)

        # CRITICAL STEP: Shift the data by 1 period. This makes each row
        # contain the data of the *previous* candle.
        df_resampled = df_resampled.shift(1)

        # Add a suffix to the column names to identify the timeframe
        df_resampled.columns = [f"{col}_{tf_name}" for col in df_resampled.columns]

        # Merge the shifted HTF data back into the original M1 DataFrame.
        # `merge_asof` is perfect for this. It finds the last available
        # (i.e., most recent) HTF candle for each M5 candle.
        df = pd.merge_asof(df, df_resampled, left_index=True, right_index=True, direction='backward')

    # 5. Clean up and return
    # The initial rows will have NaNs because there was no previous HTF candle
    df = df.dropna()
    print("Data processing complete.")
    return df


if __name__ == '__main__':
    # Define the paths in the project file.
    # Assuming they are CSV files, potentially tab-separated from MT5.
    data_files = ['DAT_MT_XAUUSD_M1_2023.csv', 'DAT_MT_XAUUSD_M1_2024.csv', 'DAT_MT_XAUUSD_M1_2025.csv']

    # Process the data
    final_df = create_multitimeframe_data(data_files)

    # Save the result to CSV as requested
    output_filename = 'XAUUSD_2023_2025_Data.csv'
    final_df.to_csv(output_filename)
    print(f"\nSuccessfully saved processed data to {output_filename}")

    # Display the result to verify
    print("\n--- Final DataFrame Head ---")
    print(final_df.head())
    print("\n--- Final DataFrame Tail ---")
    print(final_df.tail(20)) # Show more rows to see the HTF data change
