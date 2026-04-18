-- ============================================================
-- 05_profit_triage.sql
-- Campaign ROI Optimisation Engine
-- ============================================================
-- PURPOSE:
--   Connect model output scores back to business decisions.
--   This query layer sits AFTER the ML model — it takes the
--   scored_customers.csv output and answers:
--     "Given our capacity, who do we call and what do we earn?"
--
-- SOURCE TABLES:
--   scored_customers  (outputs/scored_customers.csv)
--   bank_features     (view from 01_feature_engineering.sql)
--
-- QUERIES:
--   1. Capacity scenario comparison
--   2. Profit by job segment (top N customers)
--   3. Profit by age band
--   4. Score decile analysis
--   5. High score + low conversion segment detection
--   6. Full triage table for Power BI
-- ============================================================


-- ── Query 1: Capacity scenario comparison ────────────────────
-- How does profit and ROI change at different call capacities?
-- Directly mirrors the Python capacity_sensitivity.csv output
-- but calculated in SQL for validation and transparency.
SELECT
    capacity,
    n_called,
    total_expected_profit,
    total_cost,
    ROUND(roi * 100, 2)                 AS roi_pct,
    max_achievable_profit,
    -- How much of max profit are we capturing?
    ROUND(
        100.0 * total_expected_profit / max_achievable_profit,
        2
    )                                   AS pct_of_max_captured
FROM capacity_sensitivity
ORDER BY capacity;


-- ── Query 2: Profit by job segment (top 1000 customers) ──────
-- Among the top 1,000 ranked customers, which job segments
-- contribute the most expected profit?
-- This is the "who are we actually calling?" breakdown.
SELECT
    sc.job,
    COUNT(*)                            AS customers_in_top_1000,
    ROUND(SUM(sc.expected_profit), 2)   AS total_expected_profit,
    ROUND(AVG(sc.p_subscribe) * 100, 2) AS avg_subscription_prob_pct,
    ROUND(AVG(sc.expected_profit), 2)   AS avg_expected_profit_per_customer
FROM scored_customers sc
WHERE sc.rank <= 1000
GROUP BY sc.job
ORDER BY total_expected_profit DESC;


-- ── Query 3: Profit by age band ───────────────────────────────
-- Age band breakdown within the top capacity tier.
SELECT
    CASE
        WHEN sc.age < 25 THEN 'Under 25'
        WHEN sc.age BETWEEN 25 AND 34 THEN '25-34'
        WHEN sc.age BETWEEN 35 AND 44 THEN '35-44'
        WHEN sc.age BETWEEN 45 AND 54 THEN '45-54'
        WHEN sc.age BETWEEN 55 AND 64 THEN '55-64'
        ELSE '65+'
    END                                 AS age_band,
    COUNT(*)                            AS customers_selected,
    ROUND(SUM(sc.expected_profit), 2)   AS total_expected_profit,
    ROUND(AVG(sc.p_subscribe) * 100, 2) AS avg_subscription_prob_pct
FROM scored_customers sc
WHERE sc.rank <= 1000
GROUP BY age_band
ORDER BY total_expected_profit DESC;


-- ── Query 4: Score decile analysis ───────────────────────────
-- Divide all customers into 10 equal buckets by subscription
-- probability. How does conversion and profit vary by decile?
-- Classic lift chart data — used in Power BI decile bar chart.
SELECT
    decile,
    COUNT(*)                            AS customers,
    ROUND(AVG(p_subscribe) * 100, 2)    AS avg_prob_pct,
    ROUND(MIN(p_subscribe) * 100, 2)    AS min_prob_pct,
    ROUND(MAX(p_subscribe) * 100, 2)    AS max_prob_pct,
    ROUND(SUM(expected_profit), 2)      AS total_expected_profit,
    ROUND(AVG(expected_profit), 4)      AS avg_expected_profit
FROM (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY p_subscribe DESC) AS decile
    FROM scored_customers
) deciled
GROUP BY decile
ORDER BY decile;


-- ── Query 5: Precision at K — lift table ──────────────────────
-- How many true conversions do we capture at each capacity tier?
-- Mirrors threshold_analysis.csv but built in SQL.
-- Shows the lift vs random baseline clearly.
SELECT
    ta.top_k                            AS call_capacity,
    ta.true_positives                   AS predicted_conversions,
    ta.precision                        AS precision_at_k,
    ta.recall                           AS recall_at_k,
    ta.lift_vs_random,
    -- What % of all possible conversions do we capture?
    ROUND(ta.recall * 100, 2)           AS pct_conversions_captured,
    -- Revenue potential at this capacity
    ROUND(ta.true_positives * 100.0, 2) AS estimated_revenue,
    -- Cost at this capacity (£5 per call)
    ta.top_k * 5.0                      AS total_call_cost,
    -- Net profit estimate
    ROUND((ta.true_positives * 100.0) - (ta.top_k * 5.0), 2)
                                        AS estimated_net_profit
FROM threshold_analysis ta
ORDER BY ta.top_k;


-- ── Query 6: Full triage table for Power BI ──────────────────
-- Master output table joining model scores back to customer
-- demographics. This is the PRIMARY table loaded into Power BI.
-- Save the result as: outputs/triage_master.csv
SELECT
    sc.rank,
    sc.p_subscribe,
    ROUND(sc.p_subscribe * 100, 2)      AS subscription_prob_pct,
    sc.expected_profit,
    sc.cum_profit,
    -- Risk tier based on score
    CASE
        WHEN sc.p_subscribe >= 0.5  THEN 'Tier 1 — High'
        WHEN sc.p_subscribe >= 0.25 THEN 'Tier 2 — Medium'
        WHEN sc.p_subscribe >= 0.10 THEN 'Tier 3 — Low'
        ELSE 'Tier 4 — Very Low'
    END                                 AS score_tier,
    -- Capacity flag — is this customer selected at 1k / 5k / 10k?
    CASE WHEN sc.rank <= 1000  THEN 1 ELSE 0 END AS in_top_1000,
    CASE WHEN sc.rank <= 5000  THEN 1 ELSE 0 END AS in_top_5000,
    CASE WHEN sc.rank <= 10000 THEN 1 ELSE 0 END AS in_top_10000,
    -- Demographics (joined from raw features)
    bf.age,
    bf.age_band,
    bf.job,
    bf.education,
    bf.marital,
    bf.contact,
    bf.month,
    bf.day_of_week,
    bf.campaign,
    bf.campaign_bucket,
    bf.previous_contacted,
    bf.poutcome,
    bf.euribor_band,
    bf.employment_climate,
    bf.emp_var_rate,
    bf.euribor3m,
    bf.nr_employed
FROM scored_customers sc
JOIN bank_features bf ON sc."index" = bf.rowid
ORDER BY sc.rank;