-- ============================================================
-- 03_campaign_efficiency.sql
-- Campaign ROI Optimisation Engine
-- ============================================================
-- PURPOSE:
--   Answer the operational question: how many times should
--   you call a customer before giving up?
--
--   This is the "diminishing returns" analysis. Calling the
--   same customer repeatedly wastes capacity that could be
--   used on fresh high-probability customers.
--
-- DEPENDS ON: bank_features view (01_feature_engineering.sql)
--
-- QUERIES:
--   1. Conversion rate by number of contacts (this campaign)
--   2. Conversion rate by month — seasonality
--   3. Conversion rate by day of week
--   4. Previous campaign outcome impact
--   5. First contact vs repeat contact comparison
--   6. Optimal call window — where does ROI peak?
-- ============================================================


-- ── Query 1: Conversion rate by campaign contact count ───────
-- The core diminishing returns analysis.
-- After 3 contacts, conversion rate drops sharply.
-- This directly justifies the capacity-constrained triage model.
SELECT
    campaign_bucket,
    campaign                                        AS exact_contact_count,
    COUNT(*)                                        AS total_customers,
    SUM(subscribed)                                 AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)                 AS conversion_rate_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_of_calls_pct
FROM bank_features
GROUP BY campaign_bucket, campaign
ORDER BY campaign;


-- ── Query 2: Conversion rate by month ────────────────────────
-- Strong seasonality — March, September, October, December
-- are historically the best months for this campaign.
-- Used in Power BI monthly trend line chart.
SELECT
    month,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    -- Order months correctly (not alphabetically)
    CASE month
        WHEN 'jan' THEN 1  WHEN 'feb' THEN 2
        WHEN 'mar' THEN 3  WHEN 'apr' THEN 4
        WHEN 'may' THEN 5  WHEN 'jun' THEN 6
        WHEN 'jul' THEN 7  WHEN 'aug' THEN 8
        WHEN 'sep' THEN 9  WHEN 'oct' THEN 10
        WHEN 'nov' THEN 11 WHEN 'dec' THEN 12
    END AS month_num
FROM bank_features
GROUP BY month
ORDER BY month_num;


-- ── Query 3: Conversion rate by day of week ──────────────────
-- Does the day you call matter?
SELECT
    day_of_week,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    CASE day_of_week
        WHEN 'mon' THEN 1 WHEN 'tue' THEN 2
        WHEN 'wed' THEN 3 WHEN 'thu' THEN 4
        WHEN 'fri' THEN 5
    END AS day_num
FROM bank_features
GROUP BY day_of_week
ORDER BY day_num;


-- ── Query 4: Previous campaign outcome impact ─────────────────
-- Customers with a previous SUCCESS are dramatically more
-- likely to subscribe again. This is the strongest individual
-- signal after macroeconomic features.
SELECT
    poutcome                            AS previous_outcome,
    previous_contacted,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct
FROM bank_features
GROUP BY poutcome, previous_contacted
ORDER BY conversion_rate_pct DESC;


-- ── Query 5: First contact vs repeat contact ──────────────────
-- Clean binary split — never contacted before vs already contacted.
-- Key insight: previously contacted customers convert at 2-3x rate.
SELECT
    CASE previous_contacted
        WHEN 1 THEN 'Previously contacted'
        ELSE 'First time contact'
    END AS contact_history,
    COUNT(*)                            AS total_customers,
    SUM(subscribed)                     AS total_subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    ROUND(AVG(campaign), 1)             AS avg_calls_this_campaign
FROM bank_features
GROUP BY previous_contacted
ORDER BY conversion_rate_pct DESC;


-- ── Query 6: Optimal call window ─────────────────────────────
-- At what contact count does cumulative conversion peak?
-- This is the "when to stop calling" decision table.
-- Export this as optimal_call_window.csv for Power BI.
SELECT
    campaign                            AS contact_count,
    COUNT(*)                            AS customers_at_this_count,
    SUM(subscribed)                     AS subscribed,
    ROUND(AVG(subscribed) * 100, 2)     AS conversion_rate_pct,
    -- Cumulative customers called up to this point
    SUM(COUNT(*)) OVER (ORDER BY campaign ROWS UNBOUNDED PRECEDING)
                                        AS cumulative_customers,
    -- Cumulative conversions
    SUM(SUM(subscribed)) OVER (ORDER BY campaign ROWS UNBOUNDED PRECEDING)
                                        AS cumulative_conversions,
    -- Running conversion rate
    ROUND(
        100.0 * SUM(SUM(subscribed)) OVER (ORDER BY campaign ROWS UNBOUNDED PRECEDING)
        / SUM(COUNT(*)) OVER (ORDER BY campaign ROWS UNBOUNDED PRECEDING),
        2
    )                                   AS running_conversion_rate_pct
FROM bank_features
GROUP BY campaign
ORDER BY campaign;