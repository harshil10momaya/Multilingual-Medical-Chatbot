import pandas as pd
import sys
import os

RAW_FAQ_FILE_NAME = "train.csv" 
CLEAN_FAQ_FILE_NAME = "faq_knowledgebase.csv"


print("Starting programmatic processing of raw FAQ data...")

try:
    df_raw = pd.read_csv(RAW_FAQ_FILE_NAME)
except FileNotFoundError:
    print(f"CRITICAL ERROR: Please ensure the raw FAQ data file is named '{RAW_FAQ_FILE_NAME}' and is in the 'src' folder.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR loading FAQ file: {e}")
    sys.exit(1)


# 2. Select and clean the required columns
# Assuming the column names are exactly 'Question' and 'Answer' as you specified.
if 'Question' not in df_raw.columns or 'Answer' not in df_raw.columns:
    print("CRITICAL ERROR: Raw FAQ file columns must be named 'Question' and 'Answer'. Please verify.")
    sys.exit(1)

df_clean = df_raw[['Question', 'Answer']].copy()

# 3. Basic Cleaning
df_clean['Question'] = df_clean['Question'].str.strip()
df_clean['Answer'] = df_clean['Answer'].str.strip()
df_clean = df_clean.dropna()
df_clean = df_clean.drop_duplicates(subset=['Question'])

# 4. Save the finalized, clean knowledge base
df_clean.to_csv(CLEAN_FAQ_FILE_NAME, index=False)

print(f"Successfully created clean FAQ knowledge base: {CLEAN_FAQ_FILE_NAME}")
print(f"Total usable FAQ entries: {len(df_clean)}")