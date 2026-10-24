import os
import glob
import pandas as pd

def summarize_current_directory_csvs(max_rows=5):
    """
    Scans the current working directory for all CSV files, reads the header
    and first few rows of each, and prints a clean, structured context
    summary optimized for LLM ingestion.
    """
    current_dir = os.getcwd()
    csv_files = glob.glob(os.path.join(current_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in current directory: '{current_dir}'")
        return

    print(f"=== CURRENT DIRECTORY CSV SUMMARY ({len(csv_files)} files found) ===\n")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            # Load only the required top rows for memory and speed efficiency
            df = pd.read_csv(file_path, nrows=max_rows)
            
            print(f"--- FILE: {filename} ---")
            print(f"Total Columns ({len(df.columns)}): {list(df.columns)}")
            print(f"Data Types:\n{df.dtypes.to_string()}")
            print(f"\nSample Data (First {min(len(df), max_rows)} Rows):")
            print(df.to_string(index=False))
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"--- FILE: {filename} ---")
            print(f"Error reading file: {e}\n" + "="*60 + "\n")

if __name__ == '__main__':
    summarize_current_directory_csvs()