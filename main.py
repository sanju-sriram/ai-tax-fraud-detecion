import streamlit as st
import pandas as pd
import numpy as np

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Tax Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# 1. DATA GENERATION & INITIALIZATION
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """Generates synthetic tax compliance and network dataset."""
    np.random.seed(42)
    n_samples = 100
    
    tax_ids = [f"TAX-{1000 + i}" for i in range(n_samples)]
    income = np.random.uniform(20000, 500000, n_samples)
    tax_paid = income * np.random.uniform(0.05, 0.30, n_samples)
    deductions = income * np.random.uniform(0.10, 0.60, n_samples)
    
    # Generate risk scores and graph metrics
    risk_score = np.random.uniform(0.1, 0.99, n_samples)
    graph_pagerank = np.random.uniform(0.01, 0.95, n_samples)
    anomaly_score = np.random.uniform(0.0, 1.0, n_samples)
    
    # Flag fraud for entities with high combined risk
    fraud_flag = (risk_score > 0.70) | (anomaly_score > 0.80)
    
    df = pd.DataFrame({
        "tax_id": tax_ids,
        "reported_income": np.round(income, 2),
        "tax_paid": np.round(tax_paid, 2),
        "claimed_deductions": np.round(deductions, 2),
        "risk_score": np.round(risk_score, 2),
        "graph_pagerank": np.round(graph_pagerank, 3),
        "anomaly_score": np.round(anomaly_score, 2),
        "fraud_flag": fraud_flag
    })
    return df

df = load_data()

# ---------------------------------------------------------
# 2. SIDEBAR NAVIGATION & FILTERS
# ---------------------------------------------------------
st.sidebar.title("🛡️ Fraud Engine Controls")
st.sidebar.markdown("---")

risk_threshold = st.sidebar.slider(
    "Flagging Risk Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.70, 
    step=0.05
)

search_tax_id = st.sidebar.text_input("Search Entity by Tax ID", "")

# Apply Filter
flagged_df = df[df["risk_score"] >= risk_threshold].sort_values(by="risk_score", ascending=False)

# ---------------------------------------------------------
# 3. MAIN DASHBOARD HEADER & METRICS
# ---------------------------------------------------------
st.title("🛡️ AI-Powered Tax Fraud Detection Dashboard")
st.markdown("Automated risk identification, network graph insights, and audit prioritization.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tax Entities", len(df))
col2.metric("Flagged High-Risk Entities", len(flagged_df))
col3.metric("Avg Fraud Risk Score", f"{df['risk_score'].mean():.2f}")
col4.metric("Potential Unpaid Revenue", f"${flagged_df['reported_income'].sum() * 0.15:,.2f}")

st.markdown("---")

# ---------------------------------------------------------
# 4. HIGH-RISK PRIORITY AUDIT QUEUE (FIXED SECTION)
# ---------------------------------------------------------
st.subheader("🚨 Priority Audit Queue")

if not flagged_df.empty:
    # Safely select columns to display (prevents KeyError if schema changes)
    desired_cols = ["tax_id", "risk_score", "fraud_flag", "graph_pagerank", "anomaly_score", "reported_income"]
    display_cols = [col for col in desired_cols if col in flagged_df.columns]
    
    if display_cols:
        st.dataframe(flagged_df[display_cols], use_container_width=True)
    else:
        st.dataframe(flagged_df, use_container_width=True)
else:
    st.info("No high-risk entities flagged at the selected threshold.")

st.markdown("---")

# ---------------------------------------------------------
# 5. FEATURE INSIGHTS & EXPLAINABILITY
# ---------------------------------------------------------
left_chart, right_chart = st.columns(2)

with left_chart:
    st.subheader("📊 Feature Insights: Risk vs Income")
    st.scatter_chart(
        df, 
        x="reported_income", 
        y="risk_score", 
        color="fraud_flag"
    )

with right_chart:
    st.subheader("🌐 Network Anomaly Metrics")
    st.line_chart(
        df[["risk_score", "graph_pagerank", "anomaly_score"]].sort_values("risk_score")
    )

# ---------------------------------------------------------
# 6. ENTITY EXPLAINABILITY DRILL-DOWN
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔎 Entity Explainability Inspection")

selected_id = st.selectbox(
    "Select Tax ID for In-Depth Risk Breakdown", 
    options=df["tax_id"].tolist(),
    index=0 if search_tax_id == "" or search_tax_id not in df["tax_id"].tolist() else df["tax_id"].tolist().index(search_tax_id)
)

entity_data = df[df["tax_id"] == selected_id].iloc[0]

exp_col1, exp_col2, exp_col3 = st.columns(3)
exp_col1.write(f"**Tax ID:** {entity_data['tax_id']}")
exp_col1.write(f"**Reported Income:** ${entity_data['reported_income']:,.2f}")

exp_col2.write(f"**Calculated Risk Score:** `{entity_data['risk_score']}`")
exp_col2.write(f"**Graph PageRank Score:** `{entity_data['graph_pagerank']}`")

exp_col3.write(f"**Anomaly Score:** `{entity_data['anomaly_score']}`")
exp_col3.write(f"**Audit Status:** {'🔴 High Risk' if entity_data['risk_score'] >= risk_threshold else '🟢 Clear'}")

st.progress(float(entity_data['risk_score']), text="AI Confidence Level for Fraud Risk")