import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from fpdf import FPDF
import tempfile
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Data Analysis RAG", layout="wide")

# --- CSS Dark Theme ---
dark_theme = """
    <style>
        body, .stApp {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
            font-family: 'Arial', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
        }
        [data-testid="stSidebar"] * {
            color: #e0e0e0 !important;
        }
        .sidebar-card {
            background-color: #1b1b1b;
            padding: 20px;
            margin: 10px 0;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }
        [data-testid="stFileUploader"] {
            background-color: #252525;
            border: 1px dashed #1E88E5;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        [data-testid="stFileUploader"] div {
            color: #aaa !important;
        }
        input[type="password"], input[type="text"] {
            background-color: #1e1e1e;
            color: #e0e0e0 !important;
            border: 1px solid #1E88E5;
            border-radius: 8px;
            padding: 6px;
        }
        .stButton>button {
            width: 100%;
            background-color: #1E88E5 !important;
            color: #fff !important;
            border: none;
            border-radius: 8px;
            padding: 8px;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1565C0 !important;
            transform: scale(1.02);
        }
    </style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.header("📂 Upload CSV & API Key")
    uploaded = st.file_uploader("CSV dataset", type="csv")
    openai_key = st.text_input("OpenAI API Key", type="password")
    button_api = st.button("Activate API Key")
    st.markdown('</div>', unsafe_allow_html=True)

if 'openai_key' not in st.session_state:
    st.session_state.openai_key = None
if openai_key and button_api:
    st.session_state.openai_key = openai_key
    st.sidebar.success("API Key Activated!")

# --- Helpers ---
def detect_col_type(df):
    dtypes = {}
    force_numeric = ['bounce_flag', 'revenue_$', 'conversion_flag',
                     'pages_visited', 'time_spent', 'demographic_age']
    for c in df.columns:
        series = df[c]
        if c.lower() in [f.lower() for f in force_numeric]:
            dtypes[c] = 'numeric'
            continue
        name_hint = any(x in c.lower() for x in ['date', 'timestamp'])
        looks_like_date = series.astype(str).str.contains(r'[-/:]').mean() > 0.5
        if name_hint or looks_like_date:
            try:
                parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() > len(series) * 0.5:
                    df[c] = parsed
                    dtypes[c] = 'date'
                    continue
            except:
                pass
        if pd.api.types.is_numeric_dtype(series):
            dtypes[c] = 'numeric'
        else:
            dtypes[c] = 'categorical'
    return dtypes

# --- Tests ---
def detect_test(df, grp, val):
    groups = df[grp].dropna().unique()
    vals = [df[df[grp]==g][val].dropna() for g in groups]
    normal = all(stats.shapiro(v)[1] > 0.05 for v in vals if len(v)>=3)
    eq = stats.levene(*vals)[1] > 0.05
    if len(groups)==2:
        return "t-test" if normal and eq else "mann-whitney"
    return "anova" if normal and eq else "kruskal"

def run_tests(df, grp, val):
    test = detect_test(df, grp, val)
    if test in ["t-test", "mann-whitney"]:
        g = df[grp].dropna().unique()
        a,b = df[df[grp]==g[0]][val], df[df[grp]==g[1]][val]
        if test=="t-test":
            t,p = stats.ttest_ind(a,b)
            return f"T-test: t={t:.3f}, p={p:.4f}", test
        else:
            u,p = stats.mannwhitneyu(a,b)
            return f"Mann-Whitney U={u:.3f}, p={p:.4f}", test
    elif test=="anova":
        model = ols(f'{val} ~ C({grp})', df).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        tuk = pairwise_tukeyhsd(df[val], df[grp])
        return f"ANOVA:\n{aov}\n\nTukey HSD:\n{tuk.summary()}", test
    else:
        vals = [df[df[grp]==g][val] for g in df[grp].unique()]
        h,p = stats.kruskal(*vals)
        return f"Kruskal-Wallis H={h:.3f}, p={p:.4f}", test

def run_chi2(df, col1, col2):
    tab = pd.crosstab(df[col1], df[col2])
    chi, p, _, _ = stats.chi2_contingency(tab)
    return f"Chi-Square: χ²={chi:.3f}, p={p:.4f}", chi, p

def run_regression(df, y, x):
    y_data = df[y]
    X_data = df[x]
    if not pd.api.types.is_numeric_dtype(y_data):
        le_y = LabelEncoder()
        y_data = le_y.fit_transform(y_data)
    for col in x:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le_x = LabelEncoder()
            X_data[col] = le_x.fit_transform(df[col])
    X = sm.add_constant(X_data)
    if len(np.unique(y_data)) == 2:
        model = sm.Logit(y_data, X).fit(disp=False)
    else:
        model = sm.OLS(y_data, X).fit()
    return model.summary()

# --- Plot Functions ---
def plot_trend(df, date_col, val_col, agg_func, sort_order):
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[date_col] = df[date_col].dt.floor('D')
    agg_df = df.groupby(date_col)[val_col].agg(agg_func).sort_values(
        ascending=(sort_order == "Ascending")
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=agg_df.index, y=agg_df.values, ax=ax, color="teal", marker="o")
    ax.set_title(f"Trend of {val_col} by Date ({agg_func})")
    ax.set_xlabel("Date")
    ax.set_ylabel(val_col)
    plt.xticks(rotation=45)
    return agg_df, fig

def aggregate_categorical(df, cat_col, val_col, agg_func, sort_order):
    ascending = (sort_order == "Lowest")
    agg_df = df.groupby(cat_col)[val_col].agg(agg_func).sort_values(ascending=ascending)
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=agg_df.index, y=agg_df.values, ax=ax, palette="viridis")
    ax.set_title(f"{agg_func.title()} by {cat_col}")
    ax.tick_params(axis='x', rotation=45)
    return agg_df, fig

# --- GPT Explain ---
def explain(text, question=None):
    if not st.session_state.openai_key:
        return "No API key."
    
    import os
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    
    prompt = f"Explain this statistical result for a non-technical audience:\n{text}"
    if question:
        prompt += f"\nAnswer this specific question: {question}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# --- PDF Export ---
def export_pdf(results_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Analysis Report")

    for section, content in results_dict.items():
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font("Arial", size=11)

        if isinstance(content, tuple):
            text = content[0] if len(content) > 0 else ""
            insight = content[1] if len(content) > 1 else ""
            fig = content[2] if len(content) > 2 else None

            pdf.multi_cell(0, 8, text)
            if insight:
                pdf.multi_cell(0, 8, "\nInsights:\n" + insight)
            if fig and isinstance(fig, plt.Figure):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    fig.savefig(tmp_img.name, bbox_inches="tight")
                    tmp_img.flush()
                    pdf.ln(5)
                    pdf.image(tmp_img.name, w=170)
        else:
            pdf.multi_cell(0, 8, str(content))

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name

# --- State ---
if 'results' not in st.session_state:
    st.session_state.results = {}

def save_result(section, text=None, insight=None, figure=None):
    st.session_state.results[section] = (text, insight, figure)

def show_result_inline(section):
    if section in st.session_state.results:
        text, insight, fig = (st.session_state.results[section] + (None,)*3)[:3]
        if text:
            st.text(text)
        if insight:
            st.markdown(f"**Insights:** {insight}")
        if fig and isinstance(fig, plt.Figure):
            st.pyplot(fig)

# --- MAIN ---
if uploaded:
    df = pd.read_csv(uploaded)
    st.title("📊 Data Analysis Dashboard")

    # --- Preview data 5 baris pertama ---
    st.subheader("Dataset Preview (Top 5 Rows)")
    st.dataframe(df.head(5), use_container_width=True)

    dtypes = detect_col_type(df)

    # --- Power Analysis ---
    st.markdown("---")
    st.subheader("Power Analysis (Sample Size Estimation)")
    eff = st.slider("Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.1)
    if st.button("Estimate Sample Size"):
        from statsmodels.stats.power import TTestIndPower
        n = TTestIndPower().solve_power(effect_size=eff, alpha=0.05, power=0.8)
        save_result('Power Analysis', text=f"Sample size per group: {int(np.ceil(n))}")
    show_result_inline('Power Analysis')

    # --- Statistical Analysis ---
    st.markdown("---")
    st.subheader("Statistical Analysis")
    col1 = st.selectbox("Target Column", df.columns)
    col2 = st.selectbox("Group/Predictor Column", df.columns)
    target_type = dtypes[col1]
    group_type = dtypes[col2]
    default_test = "Regression"
    if target_type == 'numeric' and group_type == 'categorical':
        n_groups = df[col2].nunique()
        default_test = "T-test / Mann-Whitney" if n_groups==2 else "ANOVA / Kruskal-Wallis"
    elif target_type == 'categorical' and group_type == 'categorical':
        default_test = "Chi-Square"

    test_choice = st.selectbox(
        "Statistical Test (auto-selected)",
        ["T-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis", "Chi-Square", "Regression"],
        index=["T-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis", "Chi-Square", "Regression"].index(default_test.split()[0])
    )
    st.info(f"Auto-selected: **{default_test}** (Target: {col1} [{target_type}], Group: {col2} [{group_type}])")

    if st.button("Run Analysis", use_container_width=True):
        if test_choice in ["T-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis"] and target_type=='numeric' and group_type=='categorical':
            res, test = run_tests(df, col2, col1)
            insight = explain(res)
            save_result('Statistical Test', text=res, insight=insight)
        elif test_choice=="Chi-Square" and target_type=='categorical' and group_type=='categorical':
            res, chi, p = run_chi2(df, col1, col2)
            insight = explain(res)
            save_result('Chi-Square Test', text=res, insight=insight)
        else:
            summary = run_regression(df, col1, [col2])
            save_result('Regression', text=str(summary))

    # Show only relevant results (no campur)
    show_result_inline('Statistical Test')
    show_result_inline('Chi-Square Test')
    show_result_inline('Regression')

    # --- Trend Analysis ---
    st.markdown("---")
    st.subheader("📈 Time-Series Trend Analysis")
    date_cols = [c for c,t in dtypes.items() if t=='date']
    if date_cols:
        dcol = st.selectbox("Select Date Column", date_cols)
        valcol = st.selectbox("Select Value Column", [c for c,t in dtypes.items() if t=='numeric'])
        agg_func = st.selectbox("Aggregation", ["mean","sum","count","max","min"])
        sort_order = st.radio("Sort Order", ["Ascending","Descending"], horizontal=True)
        if st.button("Show Trend"):
            trend_df, trend_fig = plot_trend(df, dcol, valcol, agg_func, sort_order)
            save_result('Time-Series Trend', text=trend_df.head(10).to_string(), figure=trend_fig)
        show_result_inline('Time-Series Trend')

    # --- Categorical Aggregation ---
    st.markdown("---")
    st.subheader("📊 Categorical Aggregation Analysis")
    cat_cols = [c for c,t in dtypes.items() if t=='categorical']
    if cat_cols:
        catcol = st.selectbox("Select Categorical Column", cat_cols)
        numcol = st.selectbox("Select Numeric Column for Aggregation", [c for c,t in dtypes.items() if t=='numeric'])
        agg_func = st.selectbox("Aggregation Type", ["count","sum","mean"])
        sort_order = st.radio("Sort by", ["Highest","Lowest"], horizontal=True)
        if st.button("Show Aggregation"):
            agg_df, agg_fig = aggregate_categorical(df, catcol, numcol, agg_func, sort_order)
            save_result('Category Aggregation', text=agg_df.head(10).to_string(), figure=agg_fig)
        show_result_inline('Category Aggregation')

    # --- Q&A + PDF ---
    st.markdown("---")
    st.subheader("Q & A")
    q = st.text_input("Ask about your results")
    if st.button("Get Answer") and q:
        ans = explain("Analisis dataset", q)
        save_result('AI Insights', text=ans)
    show_result_inline('AI Insights')

    if st.session_state.results:
        pdf_path = export_pdf(st.session_state.results)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Full Report (PDF)", f, file_name="analysis_report.pdf")

else:
    st.info("Upload dataset to start your analysis.")
