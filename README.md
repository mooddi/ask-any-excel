# ask-data-app (app_v2.py) — Arabic Chat with Data

A simple web app that lets anyone upload an Excel file and ask questions in Arabic to get:
- a **table** result
- an **auto chart** (when possible)
- a short **Arabic business explanation** (summary + meaning + one recommendation)

## Why this matters
Managers often have large Excel files but need quick answers without manual analysis.
This app turns Excel into fast, explainable insights.

## How it works (high level)
1) Upload Excel  
2) Convert to a dataframe/table  
3) Generate a safe SQL query (SELECT only)  
4) Execute the query locally (DuckDB)  
5) Show result + chart + Arabic explanation

## Features
- Arabic questions + Arabic explanation
- Works with different Excel schemas (different column names)
- Displays results as tables and charts
- Secrets are kept outside the code (API keys are not stored in GitHub)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app_v2.py
