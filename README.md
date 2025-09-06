# 💧 INGRES Groundwater Chatbot

An interactive **Streamlit-based chatbot dashboard** to explore groundwater data across Indian states and districts.  
It supports **English, Hindi, and Kannada** queries, including **state-level, district-level, year-wise, and district comparisons**.

---

## 🚀 Features
- 🔎 **Ask in Natural Language** – Supports English / Hindi / Kannada queries.
- 🌍 **State & District Reports** – Get groundwater availability, extraction, and categorization.
- 📊 **Interactive Visuals** – Compact bar and pie charts for groundwater stage & category.
- 🧠 **Semantic Search** – Understands queries even if names aren’t exact.
- 📅 **Year Filtering** – Fetch results for specific years (e.g., "Delhi groundwater 2024").
- ⚖️ **Comparison Mode** – Compare multiple districts side by side.
- 🌐 **Deployed on Streamlit Cloud** – No setup needed, just open the link!

---

## 📸 Demo
👉 *(Add your Streamlit app link here after deployment)*

Example Queries:
- `groundwater level in Delhi 2024`
- `compare Central and South districts 2024`
- `Bangalore groundwater status`
- `Uttar Pradesh report`

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Interactive web app
- [Pandas](https://pandas.pydata.org/) – Data processing
- [FAISS](https://github.com/facebookresearch/faiss) – Semantic search
- [SentenceTransformers](https://www.sbert.net/) – Multilingual embeddings
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) – Fuzzy matching
- [LangDetect](https://pypi.org/project/langdetect/) & [Googletrans](https://py-googletrans.readthedocs.io/) – Language detection & translation
- [Matplotlib](https://matplotlib.org/) – Data visualization

---

## 📂 Project Structure
