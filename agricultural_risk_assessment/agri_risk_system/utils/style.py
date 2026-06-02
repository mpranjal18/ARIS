from __future__ import annotations

import streamlit as st


def apply_glassmorphism_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #e7f8ed 0%, #f8fbff 45%, #ffffff 100%);
            }
            .main .block-container {
                max-width: 1240px;
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }
            h1, h2, h3, h4 {
                position: relative;
                color: #0f172a;
                letter-spacing: 0.1px;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.62);
                border: 1px solid rgba(255, 255, 255, 0.38);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 18px;
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
                padding: 1rem 1.05rem;
            }
            .hero-card {
                text-align: center;
                padding: 1.3rem 1rem;
            }
            .hero-score {
                font-size: 4rem;
                font-weight: 900;
                line-height: 1;
                color: #16a34a;
                margin-top: 0.45rem;
                margin-bottom: 0.4rem;
            }
            .pill {
                display: inline-block;
                border-radius: 999px;
                padding: 0.32rem 0.9rem;
                font-size: 0.86rem;
                font-weight: 700;
                border: 1px solid rgba(0,0,0,0.06);
            }
            .kpi-title {
                color: #6b7280;
                font-size: 0.82rem;
            }
            .kpi-value {
                color: #111827;
                font-size: 1.6rem;
                font-weight: 700;
            }
            .kpi-sub {
                color: #6b7280;
                font-size: 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
