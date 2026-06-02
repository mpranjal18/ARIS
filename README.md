# ARIS — Agricultural Risk Intelligence System

This repository contains the Agricultural Risk Assessment System (ARIS).

Quick start

1. Create a Python 3.10+ virtual environment and activate it.

2. Install dependencies:

```powershell
cd C:\Users\HP\Desktop\A_IS\agricultural_risk_assessment
py -3 -m pip install -r requirements.txt
```

3. Run the full pipeline (training + outputs):

```powershell
cd C:\Users\HP\Desktop\A_IS\agricultural_risk_assessment
py -3 main.py
```

4. Run the Streamlit dashboard:

```powershell
cd C:\Users\HP\Desktop\A_IS
py -3 -m streamlit run agricultural_risk_assessment/dashboard.py
```

Generated outputs:
- `agricultural_risk_assessment/data/risk_assessment_results.csv`
- `agricultural_risk_assessment/model_comparison_table.csv`
- `visualizations/`

Notes
- If you want me to push this repository to GitHub, ensure you have the `gh` CLI installed and authenticated (`gh auth login`).
- To stop the Streamlit server: press Ctrl+C in the terminal where it is running.

License: MIT
