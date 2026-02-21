# 📊 Campaign ROI Optimization Engine  
### Bank Marketing – Profit-Driven Decision Dashboard

---

## 🔎 Overview

A profit-driven decision dashboard built on the Bank Marketing dataset.

Instead of only predicting who subscribes, this project answers the real operational question:

> Given limited daily call capacity, which customers should we contact to maximize expected profit?

This is decision optimization under operational constraints, presented as an interactive Power BI dashboard.

---

## 🧠 Business Logic

Each customer has:
- Conversion probability
- Call cost
- Potential revenue

Expected Profit per customer:

Expected Profit = (Probability × Revenue) − Cost

Customers are ranked by expected profit.  
We then simulate different call capacities to determine total profit outcomes.

---

## 🎛️ Dashboard Features

### 1️⃣ Call Capacity Slider
Simulate operational constraints (e.g., 1K, 5K, 10K calls).

### 2️⃣ KPI Cards
- Selected Capacity
- Total Expected Profit
- Total Cost
- ROI
- Maximum Achievable Profit

### 3️⃣ Profit Curve
Cumulative expected profit by ranked customers.  
Shows diminishing returns and optimal decision threshold.

### 4️⃣ Segment Analysis
Expected profit breakdown by job category.

---

## 📁 Project Structure

data/
notebooks/
outputs/
README.md


Important outputs:
- `outputs/scored_customers.csv`
- `outputs/profit_curve.csv`

---

## ⚙️ How To Reproduce

1. Run notebook in `notebooks/`
2. Generate scored customers
3. Export CSV outputs
4. Import into Power BI
5. Build dashboard with dynamic capacity simulation

---

## 🛠 Tools Used

- Python
- Pandas
- Scikit-learn
- Power BI
- DAX

---

## 📌 Dataset

Bank Marketing Dataset  
(UCI Machine Learning Repository)

---

## 🎯 Key Insight

Most profit is captured within the top-ranked customers.  
Beyond a certain capacity threshold, marginal profit decreases while costs continue rising.

This transforms predictive modeling into actionable business optimization.
