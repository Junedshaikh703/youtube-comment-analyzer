import pandas as pd
import os

# Paths
RAW_PATH = os.path.join("data", "raw", "experiment_dataset.csv")
PROCESSED_PATH = os.path.join("data", "processed", "experiment_dataset_cleaned.csv")

# Load dataset
df = pd.read_csv(RAW_PATH)

print("Before cleaning:", len(df))

# Remove empty or NaN comments
df = df.dropna(subset=["comment_text"])

# Strip whitespace
df["comment_text"] = df["comment_text"].str.strip()

# Remove empty strings after strip
df = df[df["comment_text"] != ""]

# Remove very short comments (optional but recommended)
df = df[df["comment_text"].str.len() > 2]

print("After cleaning:", len(df))

# Ensure processed directory exists
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# Save cleaned dataset
df.to_csv(PROCESSED_PATH, index=False)

print(f"Cleaned dataset saved as {PROCESSED_PATH}")