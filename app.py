import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import shap

# -------------------------------------------------------------------
# Page Config & Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Tax Fraud & Audit Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Enterprise AI Tax Fraud & Circular Evasion Engine")
st.caption("Powered by XGBoost, Isolation Forests, NetworkX Graph Analysis, and SHAP")

# -------------------------------------------------------------------
# Data Generation & Caching Functions
# -------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=1500):
    np.random.seed(42)
    tax_ids = [f"TAX_{i:04d}" for i in range(n_samples)]
    
    declared_revenue = np.random.lognormal(mean=11.5, sigma=0.8, size=n_samples)
    bank_deposits = declared_revenue * np.random.normal(loc=1.0, scale=0.1, size=n_samples)
    claimed_expenses = declared_revenue * np.random.uniform(0.5, 0.85, size=n_samples)
    vendor_claims = claimed_expenses * np.random.normal(loc=0.98, scale=0.05, size=n_samples)
    avg_days_to_claim = np.random.randint(5, 60, size=n_samples)
    
    labels = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    
    for idx in fraud_indices:
        labels[idx] = 1
        fraud_type = np.random.choice(["under_report", "phantom_expenses", "fast_claim"])
        if fraud_type == "under_report":
            bank_deposits[idx] = declared_revenue[idx] * np.random.uniform(1.8, 3.2)
        elif fraud_type == "phantom_expenses":
            claimed_expenses[idx] = declared_revenue[idx] * 0.95
            vendor_claims[idx] = claimed_expenses[idx] * np.random.uniform(0.2, 0.4)
        elif fraud_type == "fast_claim":
            avg_days_to_claim[idx] = np.random.choice([0, 1])

    df = pd.DataFrame({
        'tax_id': tax_ids,
        'declared_revenue': declared_revenue,
        'bank_deposits': bank_deposits,
        'claimed_expenses': claimed_expenses,
        'vendor_claims': vendor_claims,
        'avg_days_to_claim': avg_days_to_claim,
        'is_fraud': labels
    })
    
    # Graph construction
    G = nx.DiGraph()
    G.add_nodes_from(tax_ids)
    num_edges = len(tax_ids) * 2
    sources = np.random.choice(tax_ids, size=num_edges)
    targets = np.random.choice(tax_ids, size=num_edges)
    for s, t in zip(sources, targets):
        if s != t:
            G.add_edge(s, t)
            
    # Inject ring loops for fraudulent nodes
    fraud_ids = [tax_ids[i] for i in fraud_indices[:10]]
    for i in range(len(fraud_ids) - 1):
        G.add_edge(fraud_ids[i], fraud_ids[i+1])
    if len(fraud_ids) > 1:
        G.add_edge(fraud_ids[-1], fraud_ids[0])
        
    pagerank = nx.pagerank(G, alpha=0.85)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    df['graph_pagerank'] = df['tax_id'].map(pagerank)
    df['in_degree'] = df['tax_id'].map(in_degree)
    df['out_degree'] = df['tax_id'].map(out_degree)
    df['degree_ratio'] = (df['in_degree'] + 1) / (df['out_degree'] + 1)
    
    return df, G

@st.cache_resource
def train_hybrid_models(df):
    X = pd.DataFrame()
    X['rev_bank_ratio'] = df['bank_deposits'] / (df['declared_revenue'] + 1e-5)
    X['expense_mismatch'] = (df['claimed_expenses'] - df['vendor_claims']) / (df['claimed_expenses'] + 1e-5)
    X['profit_margin'] = (df['declared_revenue'] - df['claimed_expenses']) / (df['declared_revenue'] + 1e-5)
    X['avg_days_to_claim'] = df['avg_days_to_claim']
    X['graph_pagerank'] = df['graph_pagerank']
    X['degree_ratio'] = df['degree_ratio']
    
    y = df['is_fraud']
    
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_model.fit(X, y)
    
    iso_model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    iso_model.fit(X)
    
    # Compute hybrid scores
    sup_prob = xgb_model.predict_proba(X)[:, 1]
    raw_anomaly = iso_model.decision_function(X)
    unsup_score = 1 - (raw_anomaly - raw_anomaly.min()) / (raw_anomaly.max() - raw_anomaly.min() + 1e-5)
    
    df['fraud_risk_score'] = 0.70 * sup_prob + 0.30 * unsup_score
    df['rev_bank_ratio'] = X['rev_bank_ratio']
    df['expense_mismatch'] = X['expense_mismatch']
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer(X)
    
    return xgb_model, iso_model, df, X, shap_vals

# Load data and train models
raw_df, G = generate_synthetic_data()
model, iso_model, scored_df, X_features, shap_values = train_hybrid_models(raw_df)

# -------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Auditor Settings")
    risk_threshold = st.slider(
        "Audit Flag Risk Threshold", 
        min_value=0.30, max_value=0.90, value=0.60, step=0.05,
        help="Entities above this hybrid score will be flagged for priority audit."
    )
    
    st.divider()
    st.markdown("### Model Architecture")
    st.markdown("- **Supervised:** XGBoost Classifier (70% weight)")
    st.markdown("- **Unsupervised:** Isolation Forest (30% weight)")
    st.markdown("- **Graph Network:** NetworkX PageRank & Degree Ratios")

# Apply Threshold
scored_df['audit_flag'] = scored_df['fraud_risk_score'] >= risk_threshold
flagged_df = scored_df[scored_df['audit_flag']].sort_values(by='fraud_risk_score', ascending=False)

# -------------------------------------------------------------------
# Key Metrics KPI Bar
# -------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Taxpayers Audited", f"{len(scored_df):,}")
col2.metric("Flagged High Risk", f"{len(flagged_df):,}", delta_color="inverse")
col3.metric("Highest Risk Score", f"{scored_df['fraud_risk_score'].max()*100:.1f}%")
col4.metric("Avg Fraud Rate Detected", f"{(len(flagged_df)/len(scored_df))*100:.1f}%")

st.divider()

# -------------------------------------------------------------------
# Main Tabs Layout
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🚨 High-Risk Audit Queue", "🕸️ Circular Shell Graph Analysis", "🔍 Single Entity SHAP Inspector"])

# --- TAB 1: AUDIT QUEUE ---
with tab1:
    st.subheader("Priority Audit Queue")
    st.dataframe(
        flagged_df[[
            'tax_id', 'fraud_risk_score', 'declared_revenue', 
            'bank_deposits', 'rev_bank_ratio', 'expense_mismatch', 'graph_pagerank'
        ]].style.format({
            'fraud_risk_score': '{:.2%}',
            'declared_revenue': '${:,.2f}',
            'bank_deposits': '${:,.2f}',
            'rev_bank_ratio': '{:.2f}x',
            'expense_mismatch': '{:.2%}',
            'graph_pagerank': '{:.5f}'
        }),
        use_container_width=True
    )

# --- TAB 2: GRAPH NETWORK ANALYSIS ---
with tab2:
    st.subheader("Transaction Network Graph (Circular Trading Detection)")
    st.markdown("Shell companies passing fake invoices in loops exhibit unusually dense node edges and high PageRank values.")
    
    top_flagged_nodes = flagged_df['tax_id'].head(15).tolist()
    subgraph = G.subgraph(top_flagged_nodes)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color="#ff4b4b", node_size=600, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, edge_color="#888888", arrows=True, arrowsize=15, ax=ax)
    nx.draw_networkx_labels(subgraph, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)
    ax.set_title("Top High-Risk Invoice Subgraph Network")
    plt.axis("off")
    st.pyplot(fig)

# --- TAB 3: SHAP EXPLAINABILITY ---
with tab3:
    st.subheader("Auditor Explainability Inspector (SHAP)")
    selected_tax_id = st.selectbox("Select Tax ID to Inspect:", flagged_df['tax_id'].tolist())
    
    selected_idx = scored_df[scored_df['tax_id'] == selected_tax_id].index[0]
    entity_score = scored_df.loc[selected_idx, 'fraud_risk_score']
    
    st.warning(f"Inspecting Entity **{selected_tax_id}** — Calculated Risk Score: **{entity_score*100:.2f}%**")
    
    fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[selected_idx], max_display=6, show=False)
    plt.title(f"SHAP Feature Contribution Waterfall for {selected_tax_id}", fontsize=11)
    st.pyplot(fig_shap)