import pandas as pd

# Input and output files
INPUT_FILE = "emails.csv"     # <-- replace with your file path
OUTPUT_FILE = "emails_for_labelstudio.csv"

# Read original CSV
df = pd.read_csv(INPUT_FILE)

# Combine subject + body into one text column
df["text"] = df.apply(lambda row: f"Subject: {row['subject']} Body: {row['body']}", axis=1)

# Add unique id
df = df.reset_index().rename(columns={"index": "id"})

# Save for Label Studio
df[["id", "text"]].to_csv(OUTPUT_FILE, index=False)

print(f"Saved processed file to {OUTPUT_FILE}")
