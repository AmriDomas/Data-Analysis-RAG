# Automated Data Analysis Dashboard (Streamlit + Python)

This project is a data analysis dashboard built with Streamlit and Python.
It helps non-technical teams explore datasets quickly, run statistical tests, and generate insights—without needing to code.

## Features
1. Power Analysis
    - Estimate required sample size per group based on effect size (Cohen’s d).
    - Useful for A/B testing and experimental design before running actual tests.
2. Automatic Statistical Testing

   Detects the right test based on your dataset:
    - T-test / Mann-Whitney for comparing two groups.
    - ANOVA / Kruskal-Wallis for comparing more than two groups.
    - Chi-Square for categorical vs categorical.
    - Regression (OLS or Logistic) for predicting outcomes.
3. Trend Analysis (Time-Series)
    - Visualizes how metrics change over time (daily aggregated).
4. Categorical Aggregation
    - Aggregates and plots summaries (count, mean, sum) per category.
5. AI-Powered Q&A
    - Ask natural language questions about your results using GPT (requires API key).
6. Export as PDF
    - All results, insights, and charts are compiled into a PDF report.
  
## How to Run

1. Clone the Repository
```bash
git clone https://github.com/AmriDomas/Data-Analysis-RAG.git
cd Data-Analysis-RAG
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Launch the Streamlit App
```bash
streamlit run streamlit_test.py
```
Upload your dataset (CSV) via the sidebar, choose target and predictor columns, and start analyzing.

## Example Use Cases
 - Marketing Teams: Validate campaign performance (A/B tests).
 - Product Analysts: Track user behavior and conversion data.
 - Data Scientists: Automate exploratory data analysis for reports.

## Files in This Repo

 - `data_analysis_dashboard_portfolio.ipynb` – Jupyter Notebook (portfolio version with explanations).
 - `run streamlit_test.py` – Streamlit dashboard (interactive).
 - `requirements.txt` – Python dependencies.
 - `README.md` – Project documentation.

## Conclusion
This dashboard speeds up exploratory analysis and basic statistical testing by combining automation, visualization, and AI-assisted insights.
It’s ideal for analysts and teams who need quick, repeatable workflows without writing custom scripts.
