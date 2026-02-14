
import pandas as pd
import glob
import os

def inspect_parquet_files(directory):
    # Get all .parquet files in the directory
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    
    if not files:
        print(f"No parquet files found in {directory}")
        return

    print(f"{'='*80}")
    print(f"Inspecting {len(files)} files in: {directory}")
    print(f"{'='*80}\n")

    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"File: {file_name}")
        print(f"{'-'*40}")
        
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            # Print basic stats
            print(f"Shape: {df.shape} (Rows, Columns)")
            
            # Check for datetime column to show range
            if 'date' in df.columns:
                date_min = df['date'].min()
                date_max = df['date'].max()
                print(f"Date Range: {date_min} to {date_max}")
            elif 'datetime' in df.columns:
                 date_min = df['datetime'].min()
                 date_max = df['datetime'].max()
                 print(f"Date Range: {date_min} to {date_max}")

            # Show memory usage
            memory = df.memory_usage(deep=True).sum() / 1024**2
            print(f"Memory: {memory:.2f} MB")
            
            print("Columns (First 5):", list(df.columns[:5]))
            print("")
            
        except Exception as e:
            print(f"Error reading {file_name}: {e}\n")

if __name__ == "__main__":
    # Path to your processed data
    data_dir = "/Users/prajyotkore/Prajyot/unified_trading_rl/data/processed"
    inspect_parquet_files(data_dir)
