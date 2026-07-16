# Agricultural Risk Assessment System

AI-powered agricultural risk intelligence platform for wheat cultivation in the Bhopal-Sehore-Ashta belt of Madhya Pradesh, India.

## Overview

This project combines climate, vegetation, and pest signals to estimate crop risk and support proactive decision-making for farmers and agri-advisory teams.

Core capabilities:
- Climate risk estimation (rainfall, temperature, humidity, soil moisture behavior)
- Pest and disease risk estimation (aphid, rust, armyworm indicators)
- Combined risk scoring and model comparison
- Interactive startup-style Streamlit dashboard with forecast and insights

## Premium Dashboard Features

The dashboard includes:
- Dark/Light theme toggle
- Glassmorphism UI with modern cards and gradients
- KPI cards with sparklines and percentage deltas
- AI Insights panel with auto-generated risk narratives
- Interactive regional map (heat + marker layers)
- 7-day forecast with dotted forecast line
- Timeline player for risk-over-time animation
- Alert center for high-risk, pest-spike, and heat stress warnings
- Farmer action panel:
  - View Recommendations
  - Download Advisory
  - Simulate Risk Reduction
- Farm Health Score with badge (Excellent / Moderate / Critical)
- Model section with comparison charts + SHAP-like explainability summary
- PDF report export

## Demo Preview (Screenshots + GIF Placeholders)

Use the following placeholders to make this repository presentation-ready on GitHub.

Suggested media folder:

```text
docs/media/
|-- dashboard-hero.png
|-- kpi-cards.png
|-- risk-map-heatlayer.png
|-- ndvi-analysis.png
|-- model-explainability.png
`-- timeline-player.gif
```

### 1) Product Hero
![Product Hero](docs/media/dashboard-hero.png)

### 2) KPI Cards + AI Insights
![KPI Cards](docs/media/kpi-cards.png)

### 3) Interactive Risk Map
![Risk Map](docs/media/risk-map-heatlayer.png)

### 4) NDVI Analysis Storyboard
![NDVI Analysis](docs/media/ndvi-analysis.png)

### 5) Model Explainability
![Model Explainability](docs/media/model-explainability.png)

### 6) Timeline Player Demo (GIF)
![Timeline Player](docs/media/timeline-player.gif)

Tip: keep screenshots at 16:9 ratio (for example, 1600x900) for consistent visual presentation.

## Project Structure

```text
agricultural_risk_assessment/
|-- config/
|-- data/
|-- models/
|-- notebooks/
|-- utils/
|-- visualization/
|-- dashboard.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10+ recommended
- Windows PowerShell (or any shell)
- pip

## Setup

From the repository root or current directory:

```powershell
cd agricultural_risk_assessment
pip install -r requirements.txt
```

## Run Commands

### 1) Run full pipeline (training + outputs)

```powershell
cd agricultural_risk_assessment
python main.py
```

### 2) Run Streamlit dashboard

```powershell
cd agricultural_risk_assessment
streamlit run dashboard.py
```

If running from the repository root:

```powershell
streamlit run agricultural_risk_assessment/dashboard.py
```

## Generated Outputs

After successful run:
- Risk results CSV: data/risk_assessment_results.csv
- Model comparison CSV: model_comparison_table.csv
- Visualization assets: visualizations/
- Trained artifacts: models/artifacts/

## Regions and Crop

- Primary regions: Bhopal, Sehore, Ashta
- Secondary regions: Raisen, Vidisha
- Crop focus: Wheat (Rabi)

## Troubleshooting

1. Error: File does not exist: dashboard.py
- Ensure you are inside agricultural_risk_assessment before running streamlit.

2. Streamlit command not found
- Use: py -3 -m streamlit run dashboard.py

3. Model runtime warnings (TensorFlow/CUDA/oneDNN)
- These are typically non-blocking informational warnings on CPU systems.

4. Dashboard cache not refreshing
- Use the in-app Refresh button or restart streamlit process.

## License

MIT License.