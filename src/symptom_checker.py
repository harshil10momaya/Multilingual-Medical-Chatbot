# File Name: symptom_checker.py

import pandas as pd
import sys
import os

# --- Configuration ---
DOWNLOADED_FILE_NAME = "Symptom2Disease.csv" 
TARGET_FILE_NAME = "symptoms_text_data.csv"

# --- Main Execution ---

print("Starting data preparation for BioBERT training...")

try:
    # 1. Load the downloaded file
    df = pd.read_csv(DOWNLOADED_FILE_NAME)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Please ensure the downloaded file is named '{DOWNLOADED_FILE_NAME}' and is in the 'src' folder.")
    sys.exit(1)

# 2. Ensure columns are correctly named and select only 'label' and 'text'
# The Symptom2Disease dataset uses 'label' and 'text' columns, which we confirm here.
if 'label' not in df.columns or 'text' not in df.columns:
    print("CRITICAL ERROR: The downloaded file does not contain 'label' and 'text' columns.")
    print("Please verify the column names in the Symptom2Disease CSV.")
    sys.exit(1)

# 3. Clean and finalize the dataset
df['label'] = df['label'].str.strip().str.lower()
final_df = df[['text', 'label']].drop_duplicates().reset_index(drop=True)

# 4. Save the file in the format expected by the trainer
final_df.to_csv(TARGET_FILE_NAME, index=False)

print(f"Successfully prepared the new training file: {TARGET_FILE_NAME}")
print(f"Total unique training examples: {len(final_df)}")