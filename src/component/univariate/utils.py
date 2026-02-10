import os, re, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------- Reproducibility --------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Cleaning helpers --------------------

def _coerce_currency_column(col: pd.Series) -> pd.Series:
    """
    Convert messy currency strings to float.
    Handles multiple embedded numbers (sums them), commas, negatives in parentheses.
    """
    def parse_cell(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        is_neg = "(" in s and ")" in s
        nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
        if not nums:
            return np.nan
        val = sum(float(n) for n in nums)
        return -val if is_neg and val > 0 else val
    return col.apply(parse_cell).astype(float)

# -------------------- Date parsing --------------------
    
def _parse_dates(series: pd.Series, dayfirst="auto") -> pd.Series:
    """
    Parse a date column robustly.
    - dayfirst=True/False forces interpretation.
    - 'auto' tries both and picks the parse with more valid timestamps.
    """
    def try_flag(flag: bool) -> pd.Series:
        return pd.to_datetime(series, dayfirst=flag, errors="coerce")
    if isinstance(dayfirst, str) and str(dayfirst).lower() == "auto":
        d1 = try_flag(True)
        d2 = try_flag(False)
        return d1 if d1.notna().sum() >= d2.notna().sum() else d2
    return try_flag(bool(dayfirst))


def _pick_date_column(df: pd.DataFrame):
    preferred = ["date"]
    for c in preferred:
        if c in df.columns: return c
    for c in df.columns:
        if "date" in str(c).lower(): return c
    # fallback: best parse success
    best, best_cnt = None, -1
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        cnt = int(parsed.notna().sum())
        if cnt > best_cnt:
            best, best_cnt = c, cnt
    return best

def _pick_target_column(df: pd.DataFrame):
    preferred = ["production_cost_total"]
    for c in preferred:
        if c in df.columns: return c
    # anything with "cost"
    for c in df.columns:
        if "cost" in str(c).lower(): return c
    # most numeric-like after coercion
    best, best_cnt = None, -1
    for c in df.columns:
        coerced = _coerce_currency_column(df[c])
        cnt = int(coerced.notna().sum())
        if cnt > best_cnt:
            best, best_cnt = c, cnt
    return best

# -------------------- Data loading & aggregation --------------------
    
def load_and_aggregate_district(
    CSV_PATH: str,
    DATE_COL: str = None,
    TARGET_COL: str = None,
    dayfirst="auto",
    debug: bool = False,
    allow_autodetect: bool = True,
):
    """
    Read CSV, (optionally auto-)locate date & target columns, parse dates, coerce target to float,
    aggregate TARGET_COL by DATE_COL (district total per day).
    Returns:
        dates:  np.ndarray of datetimes (length N)
        values: np.ndarray of shape (N, 1) float32
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    cols = list(df.columns)

    # Auto-detect columns if missing or not found
    if (DATE_COL is None or DATE_COL not in cols) and allow_autodetect:
        DATE_COL = _pick_date_column(df)
        if debug: print(f"[auto] Selected DATE_COL={DATE_COL!r}")
    if (TARGET_COL is None or TARGET_COL not in cols) and allow_autodetect:
        TARGET_COL = _pick_target_column(df)
        if debug: print(f"[auto] Selected TARGET_COL={TARGET_COL!r}")

    if DATE_COL not in df.columns:
        raise ValueError(f"DATE_COL '{DATE_COL}' not in CSV columns: {cols}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not in CSV columns: {cols}")

    n_raw = len(df)

    # Parse dates robustly
    df[DATE_COL] = _parse_dates(df[DATE_COL], dayfirst=dayfirst)
    df = df.dropna(subset=[DATE_COL]).copy()

    # Coerce target to numeric
    df[TARGET_COL] = _coerce_currency_column(df[TARGET_COL])
    threshold = df[TARGET_COL].quantile(0.99)
    df = df[df[TARGET_COL] <= threshold] # remove extreme outliers

    # Group by date and sum; clean infinities/NaNs
    series = (
        df.groupby(['school_name','meal_type',DATE_COL], as_index=True)[TARGET_COL]
          .sum(min_count=1)
          .sort_index()
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
    )

    if debug:
        print(f"[load_and_aggregate_district] rows_raw={n_raw} "
              f"| rows_after_date={len(df)} "
              f"| unique_dates={series.index.nunique()} "
              f"| first_date={getattr(series.index.min(), 'date', lambda: None)()} "
              f"| last_date={getattr(series.index.max(), 'date', lambda: None)()}")

    if len(series) == 0:
        raise ValueError(
            "After parsing and aggregating, the series is empty. "
            "Check date format/values or target column content."
        )

    dates  = series.index.values
    values = series.values.astype("float32").reshape(-1, 1)
    return dates, values, DATE_COL, TARGET_COL

# -------------------- Dataset & splitting --------------------

class TimeSeriesDataset(Dataset):
    """Sliding-window dataset: last WINDOW values -> next value."""
    def __init__(self, data_2d, window: int):
        # data_2d shape: (T, 1)
        self.data = np.asarray(data_2d, dtype="float32")
        self.window = int(window)

    def __len__(self):
        # number of (window -> next) samples
        return max(0, len(self.data) - self.window)

    def __getitem__(self, idx: int):
        X = self.data[idx : idx + self.window, 0]  # (window,)
        y = self.data[idx + self.window, 0]        # scalar
        return torch.tensor(X).float(), torch.tensor(y).float()

def safe_time_split(series_2d: np.ndarray, asplit: float, window: int) -> int:
    """
    Choose a split so BOTH train and test have at least (window+1) points,
    allowing at least one (window->next) sample on each side.
    """
    n = len(series_2d)
    low  = window + 1
    high = n - (window + 1)
    if high <= low:
        raise ValueError(
            f"Series too short (n={n}) for WINDOW={window}. Need > {2*window+2} total points."
        )
    split = int(n * asplit)
    return max(low, min(split, high))
