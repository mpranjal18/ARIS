"""Generates a simple data fusion architecture diagram as an image."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from config.config import VISUALIZATION_OUTPUT_DIR


def create_data_fusion_architecture_diagram(output_dir: str = VISUALIZATION_OUTPUT_DIR) -> str:
    """Saves an architecture diagram showing pipeline stages."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    boxes = [
        (0.02, "Weather + NDVI + Pest Data"),
        (0.26, "Feature Engineering"),
        (0.46, "Risk Target Calculation"),
        (0.69, "ML/DL Model Training"),
        (0.88, "Dashboard + Alerts"),
    ]

    for x, text in boxes:
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E6F2FF", "edgecolor": "#2F6B8A"},
            transform=ax.transAxes,
        )

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.07
        x2 = boxes[i + 1][0] - 0.07
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=2))

    path = os.path.join(output_dir, "data_fusion_architecture.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
"""
Data Fusion Architecture Diagram for Agricultural Risk Assessment System
"""

data_fusion_diagram = '''
graph TD
    A[Data Sources] --> B[Data Collection Layer]
    B --> C[Data Preprocessing Layer]
    C --> D[Feature Engineering Layer]
    D --> E[Risk Assessment Engine]
    E --> F[Risk Visualization & Forecasting Dashboard]
    
    A1[NDVI/Satellite Data] --> A
    A2[Rainfall Data] --> A
    A3[Temperature/Humidity] --> A
    A4[Pest/Disease Reports] --> A
    A5[Mandi Prices] --> A
    A6[Weather Forecast] --> A
    A7[Historical Risk Data] --> A
    
    B1[ISRO Bhuvan API] --> B
    B2[IMD/Open-Meteo API] --> B
    B3[Field Surveys] --> B
    B4[Agmarknet Portal] --> B
    B5[Government Databases] --> B
    
    C1[Data Cleaning] --> C
    C2[Normalization] --> C
    C3[Outlier Detection] --> C
    C4[Missing Value Imputation] --> C
    C5[Data Integration] --> C
    
    D1[Climate Features] --> D
    D2[Pest/Disease Features] --> D
    D3[Vegetation Indices] --> D
    D4[Economic Indicators] --> D
    D5[Temporal Features] --> D
    D6[Geospatial Features] --> D
    
    E1[Climate Risk Model] --> E
    E2[Pest/Disease Risk Model] --> E
    E3[Machine Learning Models] --> E
    E4[Hybrid Models] --> E
    E5[Ensemble Methods] --> E
    E6[Forecasting Models] --> E
    
    F1[Climate Risk Graph] --> F
    F2[Pest Risk Graph] --> F
    F3[Combined Risk Visualization] --> F
    F4[NDVI Time Series] --> F
    F5[Price Trend Graph] --> F
    F6[Forecast Dashboard] --> F
    F7[Model Performance Metrics] --> F
    F8[Integrated Risk Dashboard] --> F
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style E1 fill:#ffcdd2
    style E2 fill:#ffcdd2
    style E3 fill:#ffcdd2
    style E4 fill:#ffcdd2
    style E5 fill:#ffcdd2
    style E6 fill:#ffcdd2
'''

print("Data Fusion Architecture Diagram:")
print(data_fusion_diagram)