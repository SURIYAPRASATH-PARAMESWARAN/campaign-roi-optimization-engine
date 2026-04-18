-- ============================================================
-- 02_customer_segments.sql
-- Campaign ROI Optimisation Engine
-- ============================================================
-- PURPOSE:
--   Profile the customer base across key demographic and
--   financial dimensions. This is the "who are we calling?"
--   analysis that runs BEFORE any model scoring.
--
-- DEPENDS ON: bank_features view (01_feature_engineering.sql)
--
-- QUERIES:
--   1. Subscription rate by job
--   2. Subscription rate by age band
--   3. Subscription rate by education
--   4. Subscription rate by marital status
--   5. Loan profile of subscribers vs non-subscribers
--   6. Contact method breakdown
-- ============================================================


-- ── Query 1: Subscription rate by job ────────────────────────
-- Which job segments convert best?
-- This feeds the "Profit by Job Segment" bar chart in the dashboard.
SELECT
    job,
    COUNT(*)                                        AS total_customers,
    SUM(subscribed)                                 AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)                 AS conversion_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_of_base_pct
FROM bank_features
GROUP BY job
ORDER BY conversion_rate_pct DESC;


-- ── Query 2: Subscription rate by age band ───────────────────
-- Younger and older customers behave differently.
-- Used in Power BI age band bar chart.
SELECT
    age_band,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    ROUND(AVG(age), 1)                  AS avg_age
FROM bank_features
GROUP BY age_band
ORDER BY
    CASE age_band
        WHEN 'Under 25' THEN 1
        WHEN '25-34'    THEN 2
        WHEN '35-44'    THEN 3
        WHEN '45-54'    THEN 4
        WHEN '55-64'    THEN 5
        ELSE 6
    END;


-- ── Query 3: Subscription rate by education ──────────────────
-- Does education level predict willingness to subscribe?
SELECT
    education,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY education
ORDER BY conversion_rate_pct DESC;


-- ── Query 4: Subscription rate by marital status ─────────────
SELECT
    marital,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY marital
ORDER BY conversion_rate_pct DESC;


-- ── Query 5: Loan profile — subscribers vs non-subscribers ───
-- Do customers with existing loans subscribe less?
-- Helps the bank understand financial product cross-sell risk.
SELECT
    has_housing_loan,
    has_personal_loan,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
WHERE has_housing_loan IS NOT NULL
  AND has_personal_loan IS NOT NULL
GROUP BY has_housing_loan, has_personal_loan
ORDER BY conversion_rate_pct DESC;


-- ── Query 6: Contact method breakdown ────────────────────────
-- Telephone vs cellular — which channel converts better?
SELECT
    contact,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY contact
ORDER BY conversion_rate_pct DESC;


-- ── Query 7: Full segment summary for Power BI ───────────────
-- One clean table combining job + age band + conversion rate.
-- Load this directly into Power BI as the segment table.
SELECT
    job,
    age_band,
    education,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    COUNT(*) - SUM(subscribed)          AS total_not_subscribed
FROM bank_features
GROUP BY job, age_band, education
ORDER BY conversion_rate_pct DESC;