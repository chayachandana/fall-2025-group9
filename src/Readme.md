# Overview 

The src/ folder contains the complete backend pipeline for the FCPS Meal Production Forecasting project. It is responsible for transforming raw production records into clean datasets and training forecasting models that power both the research paper and the Streamlit dashboard.

🔧 What Happens Inside src/

1️⃣ Data Ingestion & Cleaning
- Parses breakfast and lunch HTML production records
- Extracts served, planned, discarded, leftover, and cost values
- Removes noise, fixes inconsistent formats, and standardizes all fields
- Generates combined CSVs in `Data/Output/`

2️⃣ Exploratory Data Analysis (EDA)
- Visualizes cost trends, waste ratios, and participation patterns
- Generates plots used in the dashboard & research paper
- Performs outlier detection and statistical summaries

3️⃣ Forecasting Models
- Univariate models: LSTM, GRU, FNN, XGBoost, Linear Regression
- Multivariate models: GRU, LSTM, XGBoost, FNN
- Creates sliding windows, trains models, evaluates metrics, and saves outputs

4️⃣ Pipeline Scripts
- `combine_csv.py` → HTML → CSV merging
- `univariate_main.py` → Runs univariate forecasting
- `multivariate_main.py` → Runs multivariate forecasting
- Saves plots + results to `demo/images/`

## src folder


```text
src
├── Data
│   ├── Html
│   │   ├── May 2025 Breakfast production records
│   │   │   (all daily *.html breakfast files)
│   │   └── May 2025 Lunch production records
│   │       (all daily *.html lunch files)
│   │
│   └── Output
│       ├── breakfast_combined.csv
│       ├── lunch_combined.csv
│       └── meals_combined.csv
│
├── component
│   ├── EDA.py
│   ├── preprocess.py
│   │
│   ├── multivariate
│   │   ├── model.py
│   │   ├── plot.py
│   │   ├── training.py
│   │   └── utils.py
│   │
│   └── univariate
│       ├── comparing_model.py
│       ├── forecasting.py
│       ├── model.py
│       ├── plot.py
│       ├── training.py
│       └── utils.py
│
├── maincode
│   └── main.py
│
├── results
│   └── all_school_meal_forecasts.csv
│
├── tests
│    ├── combine_csv.py
│    └── multivariate_main.py
│   
│
├── .gitkeep
└── requirements.txt
```


## 🚀 How to Run

To execute the full pipeline and generate outputs :

Ensure prerequisites are installed.

Run the Main :
```bash
python src/maincode/main.py
```

(Note: Adjust the path based on your current working directory)
