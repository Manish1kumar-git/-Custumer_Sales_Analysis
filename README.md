# AI-Powered Customer Sales & Churn Analytics

## Overview
A portfolio/college-ready Data Analyst project demonstrating how AI-assisted data cleaning,
anomaly detection, machine learning and automated business insights can be combined in one workflow.

## AI Usage
1. **AI-assisted data profiling/cleaning:** the project creates cleaning recommendations from detected data-quality problems.
2. **AI anomaly detection:** Isolation Forest flags unusual customer transactions.
3. **AI-assisted insight generation:** business KPIs and findings are converted into concise insight statements.
4. **ML prediction:** Random Forest predicts customer churn.

> For a production version, the `ai_cleaning_advisor` and `ai_business_insights` functions can be connected to an LLM API. Never hard-code an API key; use environment variables.

## Dataset
`data/raw_dataset.csv` is a synthetic customer-sales dataset with intentionally injected missing values,
duplicates, inconsistent text, invalid ages and negative sales so that the cleaning process is demonstrable.

## Run
```bash
pip install -r requirements.txt
python src/project.py
```

## Suggested dashboard
```bash
streamlit run dashboard/app.py
```

## Key KPIs
- Total Sales
- Average Order Value
- Churn Rate
- Customer Satisfaction
- AI-detected Anomalies
- Sales by Region and Category

## Project Workflow
Raw Data -> AI-assisted Profiling -> Cleaning -> AI Anomaly Detection -> EDA/KPIs -> Churn ML -> Business Insights

## Interview explanation
"I used AI-assisted profiling to identify data-quality problems and generate cleaning recommendations. I then
implemented deterministic Python rules for reproducible cleaning, used Isolation Forest for anomaly detection,
and trained a Random Forest churn model. This separates AI assistance from the final reproducible analytical pipeline."
