from __future__ import annotations

import streamlit as st


LANGUAGES = ["English", "Hindi"]

TEXT = {
    "English": {
        "app_name": "Agri Risk Intelligence System (ARIS)",
        "subtitle": "AI-powered platform analyzing multiple dimensions of crop stress",
        "subtitle_hi_line": "Kisanon ke liye smart nirnay sahayata | किसानों के लिए स्मार्ट निर्णय सहायता",
        "greet_morning": "🌅 Good Morning, Pranjal",
        "greet_afternoon": "☀️ Good Afternoon, Pranjal",
        "greet_evening": "🌙 Good Evening, Pranjal",
        "risk_score": "Risk Score",
        "climate_risk": "Climate Risk",
        "pest_risk": "Pest Risk",
        "forecast_risk": "Market Risk",
        "smart_insight": "Smart Insight",
        "combined_risk_graph": "Combined Risk Graph",
        "download_report": "Download Report",
        "language": "Language",
        "region": "Region",
        "date_range": "Date Range",
        "ndvi_page": "NDVI Intelligence",
        "pest_climate_page": "Pest & Climate Analysis",
        "market_page": "Market Intelligence",
        "settings_page": "Settings",
        "model_page": "Model Insights",
        "home_page": "Home / Dashboard",
        "live_mandi": "Live Mandi Prices",
        "current_price": "Current Price",
        "quick_actions": "Quick Actions",
        "wait_days": "📈 Wait 2 days before selling",
        "sell_now": "📉 Sell now to avoid loss",
        "alert_high": "High risk detected. Immediate field intervention advised.",
        "alert_med": "Moderate risk detected. Increase monitoring frequency.",
        "alert_low": "Low risk detected. Continue regular good practices.",
        "best_sell_text": "Best time to sell: Next 10-15 days",
    },
    "Hindi": {
        "app_name": "Agri Risk Intelligence System (ARIS)",
        "subtitle": "एआई आधारित प्लेटफ़ॉर्म जो फसल तनाव के कई आयामों का विश्लेषण करता है",
        "subtitle_hi_line": "Kisanon ke liye smart nirnay sahayata | किसानों के लिए स्मार्ट निर्णय सहायता",
        "greet_morning": "🌅 सुप्रभात, Pranjal",
        "greet_afternoon": "☀️ नमस्कार, Pranjal",
        "greet_evening": "🌙 शुभ संध्या, Pranjal",
        "risk_score": "जोखिम स्कोर",
        "climate_risk": "जलवायु जोखिम",
        "pest_risk": "कीट जोखिम",
        "forecast_risk": "बाजार जोखिम",
        "smart_insight": "स्मार्ट इनसाइट",
        "combined_risk_graph": "संयुक्त जोखिम ग्राफ",
        "download_report": "रिपोर्ट डाउनलोड करें",
        "language": "भाषा",
        "region": "क्षेत्र",
        "date_range": "तिथि सीमा",
        "ndvi_page": "एनडीवीआई इंटेलिजेंस",
        "pest_climate_page": "कीट व जलवायु विश्लेषण",
        "market_page": "मार्केट इंटेलिजेंस",
        "settings_page": "सेटिंग्स",
        "model_page": "मॉडल इनसाइट्स",
        "home_page": "होम / डैशबोर्ड",
        "live_mandi": "लाइव मंडी भाव",
        "current_price": "वर्तमान मूल्य",
        "quick_actions": "त्वरित विकल्प",
        "wait_days": "📈 बेचने से पहले 2 दिन प्रतीक्षा करें",
        "sell_now": "📉 नुकसान से बचने के लिए अभी बेचें",
        "alert_high": "उच्च जोखिम मिला। तुरंत खेत स्तर पर कार्रवाई करें।",
        "alert_med": "मध्यम जोखिम मिला। निगरानी बढ़ाएं।",
        "alert_low": "कम जोखिम। नियमित अच्छी प्रथाएं जारी रखें।",
        "best_sell_text": "बेचने का सर्वश्रेष्ठ समय: अगले 10-15 दिन",
    },
}


def ensure_language_state() -> None:
    if "aris_language" not in st.session_state:
        st.session_state["aris_language"] = "English"


def get_language() -> str:
    ensure_language_state()
    return st.session_state.get("aris_language", "English")


def set_language(lang: str) -> None:
    if lang in LANGUAGES:
        st.session_state["aris_language"] = lang


def t(key: str) -> str:
    lang = get_language()
    return TEXT.get(lang, TEXT["English"]).get(key, TEXT["English"].get(key, key))
