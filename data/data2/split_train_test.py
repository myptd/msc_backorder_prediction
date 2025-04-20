import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Training_Dataset_v2.csv.gz")

# Define features and target
X = df.iloc[:, :-1]  # All columns except the last
y = df.iloc[:, -1]   # Last column (target for stratification)

# Split dataset (80% train, 20% test, stratify by target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2025
)

# Recombine features and target
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save to compressed CSV files
train_df.to_csv("train.csv.gz", index=False, compression="gzip")
test_df.to_csv("test.csv.gz", index=False, compression="gzip")

print("Dataset split completed successfully.")
