import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from test_search import search_ingres
from langdetect import detect
import re
from datetime import datetime

# ---------------------
# Google Translate fallback
# ---------------------
try:
    from googletrans import Translator
    translator = Translator()
except ImportError:
    translator = None

# ---------------------
# Page setup
# ---------------------
st.set_page_config(page_title="💧 INGRES Groundwater Dashboard", layout="wide")
st.title("💧 INGRES Groundwater Dashboard")
st.markdown(
    "Ask groundwater questions in **English / Hindi / Kannada**. Supports **state, district, year, and comparison** queries."
)

# ---------------------
# Translation helper
# ---------------------
def translate_text(text, lang_code):
    if translator and lang_code in ["hi", "kn"]:
        try:
            return translator.translate(text, dest=lang_code).text
        except:
            return text
    return text

# ---------------------
# Extract years from query
# ---------------------
def extract_years(query):
    years = []
    ranges = re.findall(r'(\b\d{4})\s*-\s*(\d{4}\b)', query)
    for start, end in ranges:
        years.extend(list(range(int(start), int(end)+1)))
    singles = re.findall(r'\b\d{4}\b', query)
    for y in singles:
        if not any(int(y) in range(int(s), int(e)+1) for s,e in ranges):
            years.append(int(y))
    return sorted(set(years))

# ---------------------
# Responsive bar chart
# ---------------------
def plot_bar_chart(df_data, group_col, value_col, title, color, lang):
    if df_data.empty or group_col not in df_data.columns or value_col not in df_data.columns:
        st.info(translate_text("No data to display", lang))
        return
    grouped = df_data.groupby(group_col)[value_col].mean()
    if grouped.empty:
        st.info(translate_text("No data to display", lang))
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    grouped.plot(kind="bar", ax=ax, color=color, alpha=0.8, width=0.6)
    ax.set_title(translate_text(title, lang), fontsize=10)
    ax.set_ylabel(translate_text("Stage (%)", lang), fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------------------
# Responsive pie chart
# ---------------------
def plot_pie_chart(df_data, title, lang):
    if df_data.empty or "category_derived" not in df_data.columns:
        st.info(translate_text("No data to display", lang))
        return
    counts = df_data["category_derived"].value_counts()
    if counts.empty:
        st.info(translate_text("No data to display", lang))
        return
    emoji_map = {
        "Safe": "💧 Safe",
        "Semi-Critical": "⚠️ Semi-Critical",
        "Critical": "🔴 Critical",
        "Over-Exploited": "⚡ Over-Exploited"
    }
    labels = [emoji_map.get(label, label) for label in counts.index]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=90,
        textprops={'fontsize':9},
        colors=["#4CAF50", "#FFC107", "#FF5722", "#9C27B0"]
    )
    ax.set_title(translate_text(title, lang), fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------------------
# Summary helper
# ---------------------
def show_summary(df_data, region_name, lang):
    if df_data.empty:
        return translate_text(f"No data available for {region_name}.", lang)
    years = df_data["assessment_year"].str[:4].astype(int)
    year_min, year_max = years.min(), years.max()
    avg_stage = df_data["stage_of_extraction_pct_total"].mean()
    total_available = df_data["annual_extractable_resource_ham_total"].sum()
    total_extraction = df_data["annual_extraction_ham_total"].sum()
    total_remaining = df_data["remaining_groundwater"].sum()
    counts = df_data["category_derived"].value_counts().to_dict()
    cat_summary = ", ".join([f"{v} {k}" for k, v in counts.items()])
    summary_text = (
        f"📍 **{region_name} Groundwater Report** ({year_min}-{year_max}):\n"
        f"- Avg Stage of Extraction: **{avg_stage:.1f}%**\n"
        f"- Total Available: **{total_available:,.0f} HAM**\n"
        f"- Total Extraction: **{total_extraction:,.0f} HAM**\n"
        f"- Remaining: **{total_remaining:,.0f} HAM**\n"
        f"- Categorization: {cat_summary if cat_summary else 'No categories'}"
    )
    return translate_text(summary_text, lang)

# ---------------------
# User query input
# ---------------------
query = st.text_input("💬 Ask your groundwater question (English / Hindi / Kannada):")

if query:
    try:
        lang = detect(query)
    except:
        lang = "en"

    requested_years = extract_years(query)
    result = search_ingres(query, requested_years=requested_years)
    data = result.get("data", pd.DataFrame())
    region_name = result.get("region", "Region")

    if data.empty:
        st.warning(translate_text(
            f"❌ No groundwater data available for {region_name} in "
            f"{', '.join(map(str, requested_years)) if requested_years else 'selected years'}.", lang
        ))
    else:
        st.subheader(translate_text(f"Groundwater Summary: {region_name}", lang))
        st.markdown(show_summary(data, region_name, lang))
        st.dataframe(data.head(5))

        # Charts side by side
        if result.get("type") in ["state", "district"]:
            col1, col2 = st.columns(2)
            with col1:
                plot_bar_chart(data, "assessment_year", "stage_of_extraction_pct_total",
                               "Stage of Extraction Over Years", "#4CAF50", lang)
            with col2:
                plot_pie_chart(data, "Categorization Distribution", lang)
        elif result.get("type") == "compare":
            col1, col2 = st.columns(2)
            with col1:
                plot_bar_chart(data, "district", "stage_of_extraction_pct_total",
                               "Stage of Extraction Comparison", "#FF9800", lang)
            with col2:
                plot_pie_chart(data, "Category Distribution Across Compared Districts", lang)

# ---------------------
# Feedback Section
# ---------------------
st.markdown("---")
st.subheader("💬 User Feedback")
feedback = st.text_area("Share your feedback about this chatbot here:")

if st.button("Submit Feedback"):
    if feedback.strip():
        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {feedback.strip()}\n")
        st.success("✅ Thank you! Your feedback has been submitted.")
    else:
        st.warning("⚠️ Please enter some feedback before submitting.")
