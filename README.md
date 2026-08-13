AI Tax Fraud Detection Demo
===========================

This repository contains a demo Streamlit app that implements a hybrid AI tax fraud detection system using:

- Supervised classifier: XGBoost
- Unsupervised detector: Isolation Forest
- Graph features: NetworkX PageRank and degree ratios
- Explainability: SHAP

Files
- tax_fraud_demo/main.py — Streamlit demo app (synthetic data + upload support)
- app.py — existing prototype Streamlit app in the repository
- requirements.txt — project dependencies

Quick start (local)

1. Open a terminal in the project root:

```powershell
cd C:\Users\SANJANA\Downloads\fastapi-rag-nvidia
.\venv\Scripts\Activate.ps1
```

2. Install dependencies (skip if already installed):

```powershell
pip install -r requirements.txt
# or minimal:
pip install streamlit xgboost scikit-learn networkx matplotlib pandas numpy shap
```

3. Run the demo app:

```powershell
streamlit run tax_fraud_demo/main.py
```

Or run the existing prototype:

```powershell
streamlit run app.py
```

Create a GitHub repository and push

Using GitHub CLI (recommended):

```powershell
cd C:\Users\SANJANA\Downloads\fastapi-rag-nvidia
git init
git add .
git commit -m "Initial commit: tax fraud demo"
gh repo create my-tax-fraud-demo --public --source=. --remote=origin --push
```

Or manual (web):
- Create a new repository on github.com, then run:

```powershell
git init
git add .
git commit -m "Initial commit: tax fraud demo"
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

Notes
- The app uses synthetic data by default; you can upload your CSV with required columns (tax_id, declared_revenue, bank_deposits, claimed_expenses, vendor_claims, avg_days_to_claim).
- Consider adding a CI workflow (GitHub Actions) to run linting or tests before merging.
