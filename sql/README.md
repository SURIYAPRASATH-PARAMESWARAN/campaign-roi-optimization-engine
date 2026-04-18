# SQL Layer — Campaign ROI Optimisation Engine

This folder contains the full SQL analytical pipeline for the Campaign ROI
Optimisation Engine. It sits alongside the Python ML pipeline as an independent
analytical layer — demonstrating that the same business questions can be answered
in pure SQL before any model is involved.

---

## Why SQL here?

The Python pipeline answers *who to call* using ML. The SQL layer answers *why*
using plain analytical queries. Together they tell the full story:

```
Raw Data (CSV)
     │
     ├── SQL Layer          → business analysis, segmentation, macro signals
     │                         feature engineering, profit triage
     │
     └── Python Pipeline    → ML models, SHAP explainability, profit optimisation
                               Streamlit dashboard
```

A data analyst in a real bank would write these SQL queries first — before any
model — to understand the data, validate assumptions, and present findings to
stakeholders who don't read Python.

---

## Files

| File | Purpose |
|---|---|
| `01_feature_engineering.sql` | Recreates `loader.py` in SQL. Creates the `bank_features` view with encoded target, age bands, campaign buckets, euribor bands, and employment climate flags. |
| `02_customer_segments.sql` | Profiles the customer base. Conversion rate by job, age band, education, marital status, loan profile, and contact method. |
| `03_campaign_efficiency.sql` | Diminishing returns analysis. How many calls before conversion drops? Seasonality by month and day. Previous campaign outcome impact. |
| `04_macro_signal_analysis.sql` | Validates the top SHAP finding in SQL: macro conditions (euribor, employment) predict subscription better than demographics. Campaign timing signal. |
| `05_profit_triage.sql` | Connects model scores to business decisions. Capacity scenarios, profit by segment, score deciles, lift table, and the master triage table for Power BI. |

---

## How to run

These queries are written in standard SQL compatible with **DuckDB** (recommended),
SQLite, or PostgreSQL.

### Option 1 — DuckDB (recommended, fastest)

DuckDB can query CSV files directly without any setup.

```bash
pip install duckdb
```

Then open a Python script or notebook:

```python
import duckdb

con = duckdb.connect()

# Load raw data directly from CSV — no import needed
con.execute("""
    CREATE VIEW bank_raw AS
    SELECT * FROM read_csv_auto(
        'data/raw/bank-additional/bank-additional-full.csv',
        sep=';'
    )
""")

# Run feature engineering view
with open('sql/01_feature_engineering.sql') as f:
    con.execute(f.read())

# Now run any analysis query
result = con.execute("""
    SELECT job, ROUND(AVG(subscribed)*100,2) AS conversion_rate
    FROM bank_features
    GROUP BY job
    ORDER BY conversion_rate DESC
""").df()

print(result)
```

### Option 2 — DuckDB CLI

```bash
duckdb

-- Inside DuckDB shell
CREATE VIEW bank_raw AS SELECT * FROM read_csv_auto('data/raw/bank-additional/bank-additional-full.csv', sep=';');
.read sql/01_feature_engineering.sql
.read sql/02_customer_segments.sql
```

### Option 3 — SQLite

```python
import sqlite3, pandas as pd

df = pd.read_csv('data/raw/bank-additional/bank-additional-full.csv', sep=';')
df.columns = [c.replace('.', '_').replace('-', '_') for c in df.columns]

con = sqlite3.connect(':memory:')
df.to_sql('bank_raw', con, index=True, if_exists='replace')

# Then run queries from each file
```

---

## Exporting for Power BI

Query 6 in `05_profit_triage.sql` produces the master triage table.
Export it as a CSV and load into Power BI:

```python
import duckdb

con = duckdb.connect()

# Setup views
con.execute("CREATE VIEW bank_raw AS SELECT * FROM read_csv_auto('data/raw/bank-additional/bank-additional-full.csv', sep=';')")

with open('sql/01_feature_engineering.sql') as f:
    con.execute(f.read())

con.execute("CREATE VIEW scored_customers AS SELECT * FROM read_csv_auto('outputs/scored_customers.csv')")
con.execute("CREATE VIEW capacity_sensitivity AS SELECT * FROM read_csv_auto('outputs/capacity_sensitivity.csv')")
con.execute("CREATE VIEW threshold_analysis AS SELECT * FROM read_csv_auto('outputs/threshold_analysis.csv')")

# Export master triage table
with open('sql/05_profit_triage.sql') as f:
    queries = f.read()

# Run Query 6 specifically and export
triage = con.execute("""
    SELECT
        sc.rank,
        sc.p_subscribe,
        ROUND(sc.p_subscribe * 100, 2) AS subscription_prob_pct,
        sc.expected_profit,
        sc.cum_profit,
        CASE
            WHEN sc.p_subscribe >= 0.5  THEN 'Tier 1 - High'
            WHEN sc.p_subscribe >= 0.25 THEN 'Tier 2 - Medium'
            WHEN sc.p_subscribe >= 0.10 THEN 'Tier 3 - Low'
            ELSE 'Tier 4 - Very Low'
        END AS score_tier,
        CASE WHEN sc.rank <= 1000  THEN 1 ELSE 0 END AS in_top_1000,
        CASE WHEN sc.rank <= 5000  THEN 1 ELSE 0 END AS in_top_5000,
        CASE WHEN sc.rank <= 10000 THEN 1 ELSE 0 END AS in_top_10000,
        sc.age, sc.job, sc.education, sc.marital,
        sc.contact, sc.month, sc.day_of_week,
        sc.campaign, sc.poutcome, sc.euribor3m
    FROM scored_customers sc
    ORDER BY sc.rank
""").df()

triage.to_csv('outputs/triage_master.csv', index=False)
print(f"Exported {len(triage):,} rows to outputs/triage_master.csv")
```

Then in Power BI Desktop: **Get Data → Text/CSV → select `triage_master.csv`**

---

## Key SQL techniques used

| Technique | Where |
|---|---|
| `CREATE VIEW` | `01_feature_engineering.sql` |
| `CASE WHEN` bucketing | All files |
| `GROUP BY` + aggregates | All files |
| Window functions (`SUM OVER`, `NTILE`) | `03_campaign_efficiency.sql`, `05_profit_triage.sql` |
| `UNION ALL` | `04_macro_signal_analysis.sql` |
| Subqueries | `05_profit_triage.sql` |
| `JOIN` | `05_profit_triage.sql` |

---

## Dataset

[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
41,188 records · 20 features · Portuguese banking institution · Moro et al. (2014)