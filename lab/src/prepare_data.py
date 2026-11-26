
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "tickets_labelled.csv")

df["group"] = df["title"].fillna("") + "||" + df["description"].fillna("")

X = df[["title", "description", "channel"]]
y = df["priority"]
groups = df["group"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

train_df = df.iloc[train_idx].drop(columns=["group"])
test_df = df.iloc[test_idx].drop(columns=["group"])

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(test_groups)

train_df.to_csv(BASE_DIR / "data" / "train.csv", index=False)
test_df.to_csv(BASE_DIR / "data" / "test.csv", index=False)

print("Dataset split completed successfully!")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Group overlap:", len(train_groups & test_groups))
print("\nTest priority distribution:")
print(test_df["priority"].value_counts())
