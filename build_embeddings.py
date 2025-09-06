import pandas as pd
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os


def build_embeddings(csv_path, out_index, meta_out, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    print(f"[INFO] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows")

    # Load model
    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create text for embeddings
    sentences = df.apply(
        lambda row: f"{row['state']} {row['district']} {row['assessment_year']} groundwater", axis=1
    ).tolist()

    print("[INFO] Generating embeddings ...")
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Build FAISS index
    d = embeddings.shape[1]  # embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))

    # Save FAISS index
    faiss.write_index(index, out_index)
    print(f"[INFO] FAISS index saved at {out_index}")

    # Save Metadata
    meta = {}
    for i, row in df.iterrows():
        meta[str(i)] = {
            "state": str(row.get("state", "")),
            "district": str(row.get("district", "")),
            "assessment_year": str(row["assessment_year"]) if not pd.isna(row["assessment_year"]) else None,
            "stage_of_extraction": row.get("stage_of_extraction_pct_total", None),
            "categorization": row.get("category_derived", None),
            "annual_recharge": row.get("annual_recharge_ham_total", None),
            "annual_extractable_resource": row.get("annual_extractable_resource_ham_total", None),
            "annual_extraction": row.get("annual_extraction_ham_total", None),
            "net_availability_future_use": row.get("net_availability_future_use_ham_total", None)
        }

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Metadata saved at {meta_out}")
    print(f"✅ Done! Index size: {index.ntotal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from groundwater CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--out_emb", type=str, default="data/ingres_index.faiss", help="Path to save FAISS index")
    parser.add_argument("--meta_out", type=str, default="data/ingres_meta.json", help="Path to save metadata JSON")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_emb), exist_ok=True)

    build_embeddings(args.csv, args.out_emb, args.meta_out)
