import pandas as pd

# Load dataset
df = pd.read_csv("experiment_dataset.csv")

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

# Save cleaned dataset
df.to_csv("experiment_dataset_cleaned.csv", index=False)

print("Cleaned dataset saved as experiment_dataset_cleaned.csv")