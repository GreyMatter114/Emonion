import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("./dataset/tweet_emotions.csv")

# Get unique entries from a specific column
unique_entries = df["sentiment"].unique()

# Print the unique entries
print(unique_entries)
