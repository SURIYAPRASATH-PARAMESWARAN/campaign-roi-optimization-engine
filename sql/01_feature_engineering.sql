-- ============================================================
-- 01_feature_engineering.sql
-- Campaign ROI Optimisation Engine
-- ============================================================
-- PURPOSE:
--   Recreate the Python feature engineering pipeline (loader.py)
--   in pure SQL. This produces the clean analytical base table
--   that feeds both business analysis queries and the ML model.
--
-- SOURCE TABLE:  bank_raw  (bank-additional-full.csv)
-- OUTPUT VIEW:   bank_features
--
-- STEPS:
--   1. Encode binary target y → 0/1
--   2. Drop duration (target leakage — only known after call ends)
--   3. Create previous_contacted flag from pdays sentinel (999 = never)
--   4. Create age bands for segmentation
--   5. Create balance tiers (using nr.employed as proxy)
--   6. Create campaign frequency buckets
-- ============================================================

-- Step 1: Create the base feature view
CREATE VIEW IF NOT EXISTS bank_features AS
SELECT
    -- ── Target ───────────────────────────────────────────────
    CASE WHEN y = 'yes' THEN 1 ELSE 0 END AS subscribed,

    -- ── Customer demographics ─────────────────────────────────
    age,
    CASE
        WHEN age < 25 THEN 'Under 25'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 44 THEN '35-44'
        WHEN age BETWEEN 45 AND 54 THEN '45-54'
        WHEN age BETWEEN 55 AND 64 THEN '55-64'
        ELSE '65+'
    END AS age_band,

    job,
    marital,
    education,

    -- ── Financial flags ───────────────────────────────────────
    CASE WHEN default_  = 'yes' THEN 1
         WHEN default_  = 'no'  THEN 0
         ELSE NULL
    END AS has_default,

    CASE WHEN housing = 'yes' THEN 1
         WHEN housing = 'no'  THEN 0
         ELSE NULL
    END AS has_housing_loan,

    CASE WHEN loan = 'yes' THEN 1
         WHEN loan = 'no'  THEN 0
         ELSE NULL
    END AS has_personal_loan,

    -- ── Contact information ───────────────────────────────────
    contact,
    month,
    day_of_week,

    -- NOTE: duration is intentionally excluded.
    -- It is only known AFTER the call ends, so using it at
    -- prediction time constitutes target leakage.

    -- ── Campaign history ──────────────────────────────────────
    campaign,
    CASE
        WHEN campaign = 1 THEN 'First contact'
        WHEN campaign BETWEEN 2 AND 3 THEN '2-3 contacts'
        WHEN campaign BETWEEN 4 AND 6 THEN '4-6 contacts'
        ELSE '7+ contacts'
    END AS campaign_bucket,

    -- pdays: 999 means customer was never previously contacted
    CASE WHEN pdays = 999 THEN 0 ELSE 1 END AS previous_contacted,
    -- pdays itself is dropped after flag creation (same as Python)

    previous,

    -- ── Previous campaign outcome ─────────────────────────────
    poutcome,
    CASE WHEN poutcome = 'success' THEN 1 ELSE 0 END AS prev_success,

    -- ── Macroeconomic indicators ──────────────────────────────
    -- These are the top SHAP features — macro climate dominates
    "emp.var.rate"   AS emp_var_rate,
    "cons.price.idx" AS cons_price_idx,
    "cons.conf.idx"  AS cons_conf_idx,
    euribor3m,
    "nr.employed"    AS nr_employed,

    -- Euribor bands for segmentation analysis
    CASE
        WHEN euribor3m < 1.0 THEN 'Very Low (<1%)'
        WHEN euribor3m BETWEEN 1.0 AND 2.0 THEN 'Low (1-2%)'
        WHEN euribor3m BETWEEN 2.0 AND 3.5 THEN 'Medium (2-3.5%)'
        ELSE 'High (>3.5%)'
    END AS euribor_band,

    -- Employment climate bucket
    CASE
        WHEN "emp.var.rate" < -1.5 THEN 'Declining'
        WHEN "emp.var.rate" BETWEEN -1.5 AND 0 THEN 'Stable'
        ELSE 'Growing'
    END AS employment_climate

FROM bank_raw;


-- ============================================================
-- QUICK VALIDATION — run these after creating the view
-- ============================================================

-- Check row count (expect ~41,188)
-- SELECT COUNT(*) AS total_rows FROM bank_features;

-- Check target distribution (expect ~11.3% positive)
-- SELECT
--     subscribed,
--     COUNT(*) AS n,
--     ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
-- FROM bank_features
-- GROUP BY subscribed;

-- Check no duration column leaked through
-- SELECT * FROM bank_features LIMIT 5;

-- Check previous_contacted distribution
-- SELECT
--     previous_contacted,
--     COUNT(*) AS n,
--     ROUND(AVG(subscribed) * 100, 2) AS conversion_rate_pct
-- FROM bank_features
-- GROUP BY previous_contacted;