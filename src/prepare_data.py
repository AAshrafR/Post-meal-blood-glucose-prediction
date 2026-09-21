import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET, GROUP_COLUMN
from features import prepare_features


def main():
    """Create and save the cleaned dataset."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    _, _, _, _, _, cleaned_df = prepare_features(raw_df, TARGET, GROUP_COLUMN)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Saved processed dataset to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
