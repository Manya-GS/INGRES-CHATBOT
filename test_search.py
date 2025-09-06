import faiss
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# -----------------------
# Load Data + Precompute
# -----------------------
CSV_PATH = "data/ingres_central_clean_district_level.csv"
INDEX_PATH = "data/ingres_index.faiss"
META_PATH = "data/ingres_meta.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

df = pd.read_csv(CSV_PATH)

# Compute remaining groundwater
df["remaining_groundwater"] = (
    df["annual_extractable_resource_ham_total"] - df["annual_extraction_ham_total"]
)

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

# -----------------------
# Helper Functions
# -----------------------
def round_stage(val):
    return round(val, 2) if pd.notna(val) else "N/A"

def generate_summary(context_df, region_name):
    """Generate groundwater summary for internal use (optional)."""
    if context_df.empty:
        return f"❌ No groundwater data found for {region_name}."
    avg_stage = round(context_df["stage_of_extraction_pct_total"].mean(), 2)
    total_extraction = context_df["annual_extraction_ham_total"].sum()
    total_available = context_df["annual_extractable_resource_ham_total"].sum()
    total_remaining = context_df["remaining_groundwater"].sum()
    counts = context_df["category_derived"].value_counts().to_dict()
    cat_summary = ", ".join([f"{v} {k}" for k, v in counts.items()])
    return {
        "avg_stage": avg_stage,
        "total_extraction": total_extraction,
        "total_available": total_available,
        "total_remaining": total_remaining,
        "category_summary": cat_summary
    }

def semantic_search(query, top_k=5):
    emb = model.encode([query])
    distances, indices = index.search(np.array(emb).astype("float32"), top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        m = META[str(idx)]
        results.append({
            "district": m.get("district"),
            "state": m.get("state"),
            "year": m.get("assessment_year"),
            "stage": round_stage(m.get("stage_of_extraction_pct_total")),
            "remaining_groundwater": round(
                m.get("annual_extractable_resource_ham_total", 0)
                - m.get("annual_extraction_ham_total", 0), 2),
            "category": m.get("category_derived"),
            "score": float(dist),
        })
    return results

# -----------------------
# Main Query Function
# -----------------------
def search_ingres(query: str, requested_years=None):
    """
    Search groundwater data.
    requested_years: optional list of int, e.g., [2023, 2024]
    """
    query_lower = query.lower().strip()
    words = query_lower.split()

    # Extract year if mentioned in requested_years
    year_filter = requested_years if requested_years else []

    # Fuzzy match state
    states = [s.lower() for s in df["state"].unique()]
    match = process.extractOne(query_lower, states, scorer=fuzz.WRatio)
    if match and match[1] > 70:
        state_name = match[0].title()
        context_df = df[df["state"].str.lower() == state_name.lower()]
        if year_filter:
            context_df = context_df[context_df["assessment_year"].str[:4].astype(int).isin(year_filter)]
        return {"type": "state", "region": state_name, "data": context_df}

    # Fuzzy match district
    districts = [d.lower() for d in df["district"].unique()]
    match = process.extractOne(query_lower, districts, scorer=fuzz.WRatio)
    if match and match[1] > 70:
        dist_name = match[0].title()
        context_df = df[df["district"].str.lower() == dist_name.lower()]
        if year_filter:
            context_df = context_df[context_df["assessment_year"].str[:4].astype(int).isin(year_filter)]
        return {"type": "district", "region": dist_name, "data": context_df}

    # Compare multiple districts (if 'compare' in query)
    if "compare" in query_lower:
        matched_dists = []
        for d in df["district"].unique():
            if d.lower() in query_lower:
                matched_dists.append(d)
        if len(matched_dists) >= 2:
            context_df = df[df["district"].isin(matched_dists)]
            if year_filter:
                context_df = context_df[context_df["assessment_year"].str[:4].astype(int).isin(year_filter)]
            return {"type": "compare", "region": "Compared Districts", "data": context_df}

    # Fallback: semantic search
    results = semantic_search(query)
    if results:
        best = results[0]
        context_df = df[(df["district"] == best["district"]) & (df["state"] == best["state"])]
        if year_filter:
            context_df = context_df[context_df["assessment_year"].str[:4].astype(int).isin(year_filter)]
        return {"type": "semantic", "region": best["district"], "data": context_df}

    return {"type": "none", "region": "", "data": pd.DataFrame()}
