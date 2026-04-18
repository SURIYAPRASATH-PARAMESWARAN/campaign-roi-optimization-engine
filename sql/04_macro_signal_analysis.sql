-- ============================================================
-- 04_macro_signal_analysis.sql
-- Campaign ROI Optimisation Engine
-- ============================================================
-- PURPOSE:
--   Validate the key SHAP insight in pure SQL:
--   macroeconomic conditions (euribor, employment) are stronger
--   predictors of subscription than individual customer features.
--
--   The "when" of the campaign matters more than the "who".
--   When interest rates fall and employment is declining,
--   term deposits become attractive regardless of customer profile.
--
-- DEPENDS ON: bank_features view (01_feature_engineering.sql)
--
-- QUERIES:
--   1. Conversion rate by euribor band
--   2. Conversion rate by employment climate
--   3. Conversion rate by consumer confidence band
--   4. Combined macro environment score
--   5. Macro vs demographic signal strength comparison
--   6. Best and worst macro windows for campaigning
-- ============================================================


-- ── Query 1: Conversion rate by euribor band ─────────────────
-- SHAP rank 3: euribor3m is the top individual numeric feature.
-- Low rates = customers prefer term deposits over market exposure.
SELECT
    euribor_band,
    ROUND(AVG(euribor3m), 3)            AS avg_euribor,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY euribor_band
ORDER BY avg_euribor;


-- ── Query 2: Conversion rate by employment climate ───────────
-- SHAP rank 1 & 2: emp_var_rate and nr_employed dominate.
-- Declining employment = more risk-averse customers = more deposits.
SELECT
    employment_climate,
    ROUND(AVG(emp_var_rate), 3)         AS avg_emp_var_rate,
    ROUND(AVG(nr_employed), 0)          AS avg_nr_employed,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY employment_climate
ORDER BY conversion_rate_pct DESC;


-- ── Query 3: Conversion rate by consumer confidence ──────────
-- Lower confidence = customers save more = higher subscription.
SELECT
    CASE
        WHEN cons_conf_idx < -50 THEN 'Very Low (<-50)'
        WHEN cons_conf_idx BETWEEN -50 AND -40 THEN 'Low (-50 to -40)'
        WHEN cons_conf_idx BETWEEN -40 AND -30 THEN 'Medium (-40 to -30)'
        ELSE 'High (>-30)'
    END                                 AS confidence_band,
    ROUND(AVG(cons_conf_idx), 2)        AS avg_confidence,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY confidence_band
ORDER BY avg_confidence;


-- ── Query 4: Combined macro environment score ─────────────────
-- When ALL macro signals align (low rates + declining employment
-- + low confidence), what does the conversion rate look like?
-- This is the "perfect storm" analysis for campaign timing.
SELECT
    euribor_band,
    employment_climate,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    -- Flag the optimal macro window
    CASE
        WHEN euribor_band = 'Very Low (<1%)'
         AND employment_climate = 'Declining'
        THEN 'OPTIMAL — launch campaign now'
        WHEN euribor_band IN ('Very Low (<1%)', 'Low (1-2%)')
         AND employment_climate IN ('Declining', 'Stable')
        THEN 'GOOD — proceed with targeting'
        ELSE 'POOR — delay or reduce capacity'
    END                                 AS campaign_timing_signal
FROM bank_features
GROUP BY euribor_band, employment_climate
ORDER BY conversion_rate_pct DESC;


-- ── Query 5: Macro vs demographic signal strength ─────────────
-- Side by side: how much does macro environment explain vs job?
-- Shows why the model ranks macro features above demographics.

-- Macro variance (euribor bands)
SELECT
    'Macro: Euribor Band'               AS signal_type,
    euribor_band                        AS segment,
    COUNT(*)                            AS n,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY euribor_band

UNION ALL

-- Demographic variance (job)
SELECT
    'Demographic: Job'                  AS signal_type,
    job                                 AS segment,
    COUNT(*)                            AS n,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY job

ORDER BY signal_type, conversion_rate_pct DESC;


-- ── Query 6: Best macro windows — export for Power BI ────────
-- Clean summary table: month + macro snapshot + conversion rate.
-- Load this as macro_windows.csv in Power BI.
SELECT
    month,
    CASE month
        WHEN 'jan' THEN 1  WHEN 'feb' THEN 2
        WHEN 'mar' THEN 3  WHEN 'apr' THEN 4
        WHEN 'may' THEN 5  WHEN 'jun' THEN 6
        WHEN 'jul' THEN 7  WHEN 'aug' THEN 8
        WHEN 'sep' THEN 9  WHEN 'oct' THEN 10
        WHEN 'nov' THEN 11 WHEN 'dec' THEN 12
    END                                 AS month_num,
    employment_climate,
    euribor_band,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    ROUND(AVG(euribor3m), 3)            AS avg_euribor,
    ROUND(AVG(emp_var_rate), 3)         AS avg_emp_var_rate
FROM bank_features
GROUP BY month, month_num, employment_climate, euribor_band
ORDER BY month_num;