import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# Config
CSV_PATH = "data/ingres_central_clean_district_level.csv"
INDEX_PATH = "data/ingres_index.faiss"
META_PATH = "data/ingres_meta.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# Load CSV
df = pd.read_csv(CSV_PATH)

# Compute remaining groundwater
df["remaining_groundwater"] = (
    df["annual_extractable_resource_ham_total"] - df["annual_extraction_ham_total"]
)
# Load or build FAISS index
model = SentenceTransformer(MODEL_NAME)

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    # Load existing index
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
    print("FAISS index and metadata loaded.")
else:
    print("Generating FAISS index and metadata...")
    # Build corpus (district + state + year)
    corpus = (df["district"].astype(str) + ", " + df["state"].astype(str) + ", " + df["assessment_year"].astype(str)).tolist()
    embeddings = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

    # Create metadata
    META = {}
    for i, row in df.iterrows():
        META[i] = {
            "district": row["district"],
            "state": row["state"],
            "assessment_year": row["assessment_year"],
            "stage_of_extraction_pct_total": row.get("stage_of_extraction_pct_total", None),
            "annual_extractable_resource_ham_total": row.get("annual_extractable_resource_ham_total", None),
            "annual_extraction_ham_total": row.get("annual_extraction_ham_total", None),
            "category_derived": row.get("category_derived", None),
        }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {META_PATH}")

# Helper functions
def round_stage(val):
    return round(val, 2) if pd.notna(val) else "N/A"

def semantic_search(query, top_k=TOP_K):
    emb = model.encode([query]).astype("float32")
    distances, indices = index.search(np.array(emb), top_k)
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
                - m.get("annual_extraction_ham_total", 0), 2
            ),
            "category": m.get("category_derived"),
            "score": float(dist),
        })
    return results

def search_ingres(query: str, requested_years=None):
    """
    Search groundwater data.
    requested_years: optional list of int, e.g., [2023, 2024]
    """
    query_lower = query.lower().strip()
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

    # Compare multiple districts if 'compare' in query
    if "compare" in query_lower:
        matched_dists = [d for d in df["district"].unique() if d.lower() in query_lower]
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

if __name__ == "__main__":
    query = "Bangalore"
    result = search_ingres(query)
    print(result["type"], result["region"])
    print(result["data"].head())
