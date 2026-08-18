import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import shap

st.set_page_config(
    page_title="AI Tax Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Tax Fraud Detection System")
st.markdown(
    "Use this demo to explore hybrid fraud detection for tax entities: supervised classification, anomaly detection, "
    "and graph-based circular transaction profiling. Upload your own dataset or analyze synthetic tax data."
)

REQUIRED_COLUMNS = [
    "tax_id",
    "declared_revenue",
    "bank_deposits",
    "claimed_expenses",
    "vendor_claims",
    "avg_days_to_claim",
]


@st.cache_data
def generate_synthetic_data(n_entities=1200, fraud_rate=0.08):
    np.random.seed(42)
    tax_ids = [f"TAX_{i:05d}" for i in range(n_entities)]

    declared_revenue = np.random.lognormal(mean=11.3, sigma=0.8, size=n_entities)
    bank_deposits = declared_revenue * np.random.normal(loc=1.0, scale=0.12, size=n_entities)
    claimed_expenses = declared_revenue * np.random.uniform(0.45, 0.85, size=n_entities)
    vendor_claims = claimed_expenses * np.random.normal(loc=0.97, scale=0.06, size=n_entities)
    avg_days_to_claim = np.random.randint(2, 60, size=n_entities)

    labels = np.zeros(n_entities, dtype=int)
    fraud_indices = np.random.choice(n_entities, size=int(n_entities * fraud_rate), replace=False)
    labels[fraud_indices] = 1

    for idx in fraud_indices:
        fraud_type = np.random.choice(["under_report", "phantom_expense", "circular_trade"])
        if fraud_type == "under_report":
            bank_deposits[idx] = declared_revenue[idx] * np.random.uniform(1.8, 3.2)
        elif fraud_type == "phantom_expense":
            claimed_expenses[idx] = declared_revenue[idx] * np.random.uniform(0.93, 0.99)
            vendor_claims[idx] = claimed_expenses[idx] * np.random.uniform(0.12, 0.35)
        elif fraud_type == "circular_trade":
            avg_days_to_claim[idx] = np.random.choice([0, 1, 2])
            vendor_claims[idx] = claimed_expenses[idx] * np.random.uniform(0.95, 0.99)

    df = pd.DataFrame(
        {
            "tax_id": tax_ids,
            "declared_revenue": declared_revenue,
            "bank_deposits": bank_deposits,
            "claimed_expenses": claimed_expenses,
            "vendor_claims": vendor_claims,
            "avg_days_to_claim": avg_days_to_claim,
            "is_fraud": labels,
        }
    )
    return df


@st.cache_data
def inject_graph_features(df: pd.DataFrame, use_real_edges: bool = False) -> pd.DataFrame:
    df = df.copy()
    nodes = df["tax_id"].tolist()
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    if use_real_edges and "source_tax_id" in df.columns and "target_tax_id" in df.columns:
        edges = df[["source_tax_id", "target_tax_id"]].dropna().astype(str).values.tolist()
        G.add_edges_from(edges)
    else:
        n_edges = max(1, len(nodes) * 2)
        sources = np.random.choice(nodes, size=n_edges)
        targets = np.random.choice(nodes, size=n_edges)
        for source, target in zip(sources, targets):
            if source != target:
                G.add_edge(source, target)

    if "is_fraud" in df.columns:
        fraud_nodes = df.loc[df["is_fraud"] == 1, "tax_id"].tolist()
    else:
        fraud_nodes = df["tax_id"].sample(min(10, len(df))).tolist()

    for i in range(len(fraud_nodes) - 1):
        G.add_edge(fraud_nodes[i], fraud_nodes[i + 1])
    if len(fraud_nodes) > 1:
        G.add_edge(fraud_nodes[-1], fraud_nodes[0])

    pagerank = nx.pagerank(G, alpha=0.85)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    df["graph_pagerank"] = df["tax_id"].map(pagerank)
    df["in_degree"] = df["tax_id"].map(in_degree)
    df["out_degree"] = df["tax_id"].map(out_degree)
    df["degree_ratio"] = (df["in_degree"] + 1) / (df["out_degree"] + 1)
    return df


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame()
    X["rev_bank_ratio"] = df["bank_deposits"] / (df["declared_revenue"] + 1e-5)
    X["expense_mismatch"] = (df["claimed_expenses"] - df["vendor_claims"]) / (
        df["claimed_expenses"] + 1e-5
    )
    X["profit_margin"] = (df["declared_revenue"] - df["claimed_expenses"]) / (
        df["declared_revenue"] + 1e-5
    )
    X["avg_days_to_claim"] = df["avg_days_to_claim"]
    X["graph_pagerank"] = df["graph_pagerank"]
    X["degree_ratio"] = df["degree_ratio"]
    return X


@st.cache_resource
def train_models(X: pd.DataFrame, y: pd.Series):
    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb_model.fit(X, y)

    iso_model = IsolationForest(n_estimators=120, contamination=0.08, random_state=42)
    iso_model.fit(X)

    return xgb_model, iso_model


def score_dataset(df: pd.DataFrame, model, iso_model, X: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["supervised_score"] = model.predict_proba(X)[:, 1]
    iso_raw = iso_model.decision_function(X)
    df["unsupervised_score"] = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min() + 1e-9)
    df["fraud_risk_score"] = 0.70 * df["supervised_score"] + 0.30 * df["unsupervised_score"]
    return df


def validate_upload(uploaded_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    messages = []
    df = uploaded_df.copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required columns: {', '.join(missing)}. "
            "Include tax_id, declared_revenue, bank_deposits, claimed_expenses, vendor_claims, avg_days_to_claim."
        )

    if "tax_id" not in df.columns:
        df["tax_id"] = [f"ROW_{i:05d}" for i in range(len(df))]
        messages.append("No tax_id column found; assigned synthetic row IDs.")

    df = df.drop_duplicates(subset=["tax_id"]).reset_index(drop=True)
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
        messages.append("No is_fraud label present in upload; the dataset will be scored as unlabeled data.")

    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].astype(float, errors="ignore")
    return df, messages


def plot_graph(df: pd.DataFrame, top_entities: list[str]):
    st.markdown("### ⚠️ Potential Circular Trading Graph")
    graph = nx.DiGraph()
    edges = []
    if "source_tax_id" in df.columns and "target_tax_id" in df.columns:
        edges = df[["source_tax_id", "target_tax_id"]].dropna().astype(str).values.tolist()
    else:
        n_nodes = len(top_entities)
        for source, target in zip(top_entities, top_entities[1:] + top_entities[:1]):
            if source != target:
                edges.append((source, target))
    graph.add_nodes_from(top_entities)
    graph.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(graph, seed=24)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="#f14c4c", ax=ax)
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=16, edge_color="#555555", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="white", ax=ax)
    ax.set_axis_off()
    st.pyplot(fig)


with st.sidebar:
    st.header("Settings")
    data_source = st.radio(
        "Dataset source",
        ["Synthetic demo data", "Upload CSV file"],
        index=0,
    )
    sample_size = st.slider("Synthetic sample size", min_value=400, max_value=2400, value=1200, step=200)
    risk_threshold = st.slider("Risk threshold", min_value=0.20, max_value=0.95, value=0.60, step=0.05)
    st.markdown("---")
    st.markdown("### Model architecture")
    st.markdown("- Supervised classifier: XGBoost")
    st.markdown("- Unsupervised detector: Isolation Forest")
    st.markdown("- Graph-based features: PageRank + degree ratios")
    st.markdown("- Explainability: SHAP feature contributions")

uploaded_file = st.sidebar.file_uploader("Upload tax entity CSV", type=["csv"])

if data_source == "Upload CSV file":
    if uploaded_file is None:
        st.warning("Upload a CSV file to score your own tax entity dataset.")
        st.stop()
    try:
        input_df = pd.read_csv(uploaded_file)
        input_df, upload_messages = validate_upload(input_df)
    except Exception as exc:
        st.error(f"Upload error: {exc}")
        st.stop()
else:
    input_df = generate_synthetic_data(sample_size)
    upload_messages = ["Using synthetic training/demo data."]

for msg in upload_messages:
    st.info(msg)

input_df = inject_graph_features(input_df, use_real_edges=("source_tax_id" in input_df.columns and "target_tax_id" in input_df.columns))
feature_matrix = create_feature_matrix(input_df)
model, iso_model = train_models(feature_matrix, input_df["is_fraud"])
scored_df = score_dataset(input_df, model, iso_model, feature_matrix)
scored_df["audit_flag"] = scored_df["fraud_risk_score"] >= risk_threshold
flagged_df = scored_df[scored_df["audit_flag"]].sort_values(by="fraud_risk_score", ascending=False)
col1.metric("Total Entities", f"{len(scored_df):,}")
col2.metric("High-Risk Flags", f"{len(flagged_df):,}")
col3.metric("Max Risk Score", f"{scored_df['fraud_risk_score'].max():.2%}")
col4.metric("Average Risk", f"{scored_df['fraud_risk_score'].mean():.2%}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🚨 High-Risk Queue", "📈 Feature Insights", "🔍 Entity Explainability"])

with tab1:
    st.write("### Priority audit queue")
    st.dataframe
        st.dataframe(flagged_df)
        use_container_width=True,
    if len(flagged_df) > 1:
        plot_graph(scored_df, flagged_df["tax_id"].head(12).tolist())

with tab2:
    st.write("### Feature distribution and model behavior")
    col_a, col_b = st.columns(2)
    col_a.metric("Supervised AUC", f"{roc_auc_score(scored_df['is_fraud'], scored_df['supervised_score']):.3f}")
    col_b.metric("Isolation Forest avg score", f"{scored_df['unsupervised_score'].mean():.3f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(scored_df['rev_bank_ratio'], scored_df['fraud_risk_score'], c=scored_df['is_fraud'], cmap='coolwarm', alpha=0.6)
    ax2.set_xlabel('Revenue / Bank Deposits Ratio')
    ax2.set_ylabel('Fraud Risk Score')
    ax2.set_title('Risk score vs revenue-bank ratio')
    st.pyplot(fig2)

    st.write("#### Top features by average value")
    st.bar_chart(
        scored_df[
            ['rev_bank_ratio', 'expense_mismatch', 'profit_margin', 'graph_pagerank', 'degree_ratio']
        ].mean()
    )

with tab3:
    st.write("### Explain why the model flagged an entity")
    selected_tax_id = st.selectbox("Choose entity to inspect", scored_df['tax_id'].tolist())
    selected_row = scored_df[scored_df['tax_id'] == selected_tax_id].iloc[0]
    st.success(
        f"{selected_tax_id} risk score={selected_row['fraud_risk_score']:.2%} | flagged={selected_row['audit_flag']}"
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(feature_matrix)7

    selected_index = scored_df.index[scored_df['tax_id'] == selected_tax_id][0]
    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[selected_index], max_display=6, show=False)
    st.pyplot(fig_shap)

    st.write("#### Entity feature profile")
    display_df = pd.DataFrame(
        {
            'feature': ['rev_bank_ratio', 'expense_mismatch', 'profit_margin', 'graph_pagerank', 'degree_ratio'],
            'value': [
                selected_row['rev_bank_ratio'],
                selected_row['expense_mismatch'],
                selected_row['profit_margin'],
                selected_row['graph_pagerank'],
                selected_row['degree_ratio'],
            ],
        }
    )
    st.table(display_df)