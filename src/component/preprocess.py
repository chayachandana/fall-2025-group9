#%%

"""Parse FCPS production HTML (many schools per file) into CSVs.

- Reads every HTML in the breakfast and lunch folders.
- Detects each school section and its table.
- Uses date from the filename.
- Keeps $ on cost columns and % on percent columns.
- Adds `meal_type` as the FIRST column ("breakfast" or "lunch").
- Writes breakfast, lunch, and a combined CSV.
"""

from pathlib import Path
from typing import Optional, Any, List
import re
import math
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ========= CONFIG (edit these 3 paths) =========
BREAKFAST_PATH = Path("src/Data/Html/May 2025 Breakfast production records")
LUNCH_PATH     = Path("src/Data/Html/May 2025 Lunch production records")
OUTDIR         = Path("src/Data/Output")

BASE_ORDER = [
    "school_name","date","identifier","name",
    "planned_reimbursable","planned_non_reimbursable","planned_total",
    "offered_total",
    "served_reimbursable","served_non_reimbursable","served_total","served_cost",
    "discarded_total","discarded_percent_of_offered","discarded_cost","subtotal_cost",
    "left_over_total","left_over_percent_of_offered","left_over_cost",
    "production_cost_total",
]

FINAL_ORDER = BASE_ORDER + ["meal_type"]

# numeric columns to parse internally
NUMERIC_COLS = [
    "planned_reimbursable","planned_non_reimbursable","planned_total",
    "offered_total",
    "served_reimbursable","served_non_reimbursable","served_total","served_cost",
    "discarded_total","discarded_percent_of_offered","discarded_cost","subtotal_cost",
    "left_over_total","left_over_percent_of_offered","left_over_cost",
    "production_cost_total",
]

CURRENCY_COLS = ["served_cost","discarded_cost","subtotal_cost","left_over_cost","production_cost_total"]
PERCENT_COLS  = ["discarded_percent_of_offered","left_over_percent_of_offered"]

CANDIDATE_HEADER_TAGS = ["h1","h2","h3","h4","strong","div","span","p"]

def list_html(folder: Path):
    """Return all .html/.htm files under a folder."""
    if not folder or not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".html", ".htm"}])

def parse_filename_date(name: str):
    """Extract date from filename like 5.01.25 → 2025-05-01."""
    m = re.search(r"(\d{1,2})[.\-_/](\d{1,2})[.\-_/](\d{2,4})", name)
    if not m: return None
    mm, dd, yy = map(int, m.groups())
    if yy < 100: yy += 2000
    try: return datetime(yy, mm, dd).strftime("%Y-%m-%d")
    except ValueError: return None

def flatten_cols(cols):
    """Flatten MultiIndex headers to single strings."""
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            flat.append(" - ".join([str(x) for x in c if str(x).strip()]).strip())
        else:
            flat.append(str(c).strip())
    return flat

def standardize(col: str):
    """Map messy header text to our canonical column names."""
    c = str(col).lower().strip().replace("–","-").replace("—","-")
    c = re.sub(r"\s+", " ", c)
    if "identifier" in c or c == "id": return "identifier"
    if c == "name" or (("name" in c or "description" in c or "menu item" in c) and "percent" not in c): return "name"
    if "planned" in c and "reimbursable" in c and "non" not in c: return "planned_reimbursable"
    if "planned" in c and ("non-reimbursable" in c or "non reimbursable" in c): return "planned_non_reimbursable"
    if "planned" in c and "total" in c: return "planned_total"
    if c.startswith("offered"): return "offered_total"
    if "served" in c and "reimbursable" in c and "non" not in c: return "served_reimbursable"
    if "served" in c and ("non-reimbursable" in c or "non reimbursable" in c): return "served_non_reimbursable"
    if "served" in c and "total" in c: return "served_total"
    if "served" in c and "cost" in c: return "served_cost"
    if "discarded" in c and "total" in c: return "discarded_total"
    if "discarded" in c and "percent" in c: return "discarded_percent_of_offered"
    if "discarded" in c and "cost" in c: return "discarded_cost"
    if "subtotal" in c: return "subtotal_cost"
    if ("left over" in c or "leftover" in c) and "total" in c: return "left_over_total"
    if ("left over" in c or "leftover" in c) and "percent" in c: return "left_over_percent_of_offered"
    if ("left over" in c or "leftover" in c) and "cost" in c: return "left_over_cost"
    if "production cost" in c: return "production_cost_total"
    if c == "cost": return "served_cost"
    return c.replace(" ", "_")

def table_looks_like_items(df: pd.DataFrame) -> bool:
    cols = [s.lower() for s in flatten_cols(df.columns)]
    return any("identifier" in c for c in cols) and any("name" in c for c in cols)

def format_currency(x):
    return "" if pd.isna(x) else f"${float(x):,.2f}"

def format_percent(x):
    return "" if pd.isna(x) else f"{float(x):.2f}%"

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one school's table and return in our column shape."""
    df = df.copy()
    df.columns = flatten_cols(df.columns)
    df.columns = [standardize(c) for c in df.columns]
    if "offered" in df.columns and "offered_total" not in df.columns:
        df = df.rename(columns={"offered": "offered_total"})
    if "cost" in df.columns and "served_cost" not in df.columns:
        df = df.rename(columns={"cost": "served_cost"})
    for c in BASE_ORDER:
        if c not in df.columns:
            df[c] = None
    df = df[~df["name"].isna()]
    df = df[df["name"].astype(str).str.strip().str.lower().ne("total")]
    df["identifier"] = df["identifier"].astype(str).str.extract(r"(\d+)", expand=False)
    df["name"] = (
        df["name"].astype(str)
        .str.replace(r"\s*\((?:Each|Package|Sandwich|Cereal Cup|Parfait)\)\s*", "", regex=True)
        .str.strip()
    )
    # numeric cleanup ($/% → numbers)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(
            df[c].astype(str)
              .str.replace(r"[\$,]", "", regex=True)
              .str.replace("%", "", regex=False)
              .str.strip()
              .replace({"": None, "nan": None}),
            errors="coerce",
        )
    # re-apply symbols
    for c in CURRENCY_COLS:
        df[c] = df[c].apply(format_currency)
    for c in PERCENT_COLS:
        df[c] = df[c].apply(format_percent)
    return df[BASE_ORDER].reset_index(drop=True)

def is_school_heading_text(text: str) -> bool:
    """True if a heading string looks like a school name."""
    if not text: return False
    if "Fairfax County Public Schools" in text: return False
    return bool(re.search(r"\b(Elementary|Middle|High|Center|Academy|School)\b", text, re.I))

def iter_tables_between(start_tag, stop_tag):
    el = start_tag
    while True:
        el = el.next_element
        if el is None or el is stop_tag:
            break
        if getattr(el, "name", None) == "table":
            yield el

def parse_file_many_schools(path: Path) -> pd.DataFrame:
    """Parse one HTML containing many schools; return stacked item rows."""
    html = path.read_text(encoding="utf-8", errors="ignore")
    date_iso = parse_filename_date(path.name)
    soup = BeautifulSoup(html, "lxml")
    headings = [t for t in soup.find_all(CANDIDATE_HEADER_TAGS) if is_school_heading_text(t.get_text(" ", strip=True))]
    frames = []
    for idx, h in enumerate(headings):
        school = re.sub(r"\s+", " ", h.get_text(" ", strip=True)).strip()
        stop = headings[idx + 1] if idx + 1 < len(headings) else None
        chosen_df = None
        for tbl in iter_tables_between(h, stop):
            try:
                dfs = pd.read_html(str(tbl))
            except Exception:
                continue
            for cand in dfs:
                if table_looks_like_items(cand):
                    chosen_df = cand
                    break
            if chosen_df is not None:
                break
        if chosen_df is None:
            continue
        norm = normalize_frame(chosen_df)
        norm["school_name"] = school
        norm["date"] = date_iso
        frames.append(norm)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def parse_folder(folder: Path, meal_type: str) -> pd.DataFrame:
    """Parse a folder and attach meal_type='breakfast' or 'lunch' (FIRST column)."""
    if not folder.exists(): return pd.DataFrame()
    frames = []
    for p in list_html(folder):
        df = parse_file_many_schools(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "meal_type", "breakfast" if meal_type.lower() == "breakfast" else "lunch")
    # ensure exact order
    for c in FINAL_ORDER:
        if c not in out.columns:
            out[c] = None
    out = out[FINAL_ORDER]
    return out

# -------- RUN --------
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
    # quick sanity print
    print("Rows by meal_type:")
    print(all_df["meal_type"].value_counts(dropna=False).to_string())
    print(f"\nSaved combined rows = {len(all_df)}")
else:
    print("No rows parsed. Check folder paths.")

# %%
