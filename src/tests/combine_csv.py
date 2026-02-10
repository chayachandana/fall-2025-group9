
from pathlib import Path
import pandas as pd
from src.component.preprocess import parse_folder, BREAKFAST_PATH, LUNCH_PATH, OUTDIR

def run_preprocessing():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    breakfast_df = parse_folder(BREAKFAST_PATH, "breakfast")
    lunch_df     = parse_folder(LUNCH_PATH,     "lunch")

    if not breakfast_df.empty:
        breakfast_df.to_csv(OUTDIR / "breakfast_combined.csv", index=False, encoding="utf-8")
    if not lunch_df.empty:
        lunch_df.to_csv(OUTDIR / "lunch_combined.csv", index=False, encoding="utf-8")

    if not breakfast_df.empty or not lunch_df.empty:
        all_df = pd.concat([x for x in [breakfast_df, lunch_df] if not x.empty], ignore_index=True)
        all_df.to_csv(OUTDIR / "meals_combined.csv", index=False, encoding="utf-8")
        print("Rows by meal_type:")
        print(all_df["meal_type"].value_counts(dropna=False).to_string())
        print(f"\nSaved combined rows = {len(all_df)}")
    else:
        print("No rows parsed. Check folder paths.")

if __name__ == "__main__":
    run_preprocessing()
