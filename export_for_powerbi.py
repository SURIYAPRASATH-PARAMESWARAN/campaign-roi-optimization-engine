import duckdb
import pandas as pd

con = duckdb.connect()

# Load raw data
con.execute("""
    CREATE VIEW bank_raw AS
    SELECT * FROM read_csv_auto(
        'data/raw/bank-additional/bank-additional-full.csv',
        sep=';'
    )
""")

# Load model outputs
sc = pd.read_csv('outputs/scored_customers.csv')
con.register('scored_customers', sc)

ta = pd.read_csv('outputs/threshold_analysis.csv')
con.register('threshold_analysis', ta)

cs = pd.read_csv('outputs/capacity_sensitivity.csv')
con.register('capacity_sensitivity', cs)

# Export triage master table
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
print(f"Done — exported {len(triage):,} rows to outputs/triage_master.csv")