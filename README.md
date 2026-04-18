# 🎵 MoodTunes AI — Music Mood Classifier & Song Recommender

> AI-powered music mood detection and song recommendation system built with LightGBM, cosine similarity, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-purple?style=flat-square)

**Module:** UFCE3P-30-3 — Essentials and Applications of Artificial Intelligence  
**Group 3 | UWE Bristol | April 2026**

---

## 🌐 Live Demo

> **Streamlit App:** [YOUR_STREAMLIT_LINK_HERE](YOUR_STREAMLIT_LINK_HERE)  
> *(Replace with your deployed Streamlit URL)*

---

## 👥 Team

| Name | Role | Key Contributions |
|---|---|---|
| **Hein Htet Phyo** | AI Product Developer | Project concept, XGBoost baseline, top-5 recommender, Streamlit UI, GitHub, final report |
| **Zulfiqar Khan** | Lead ML Engineer | Extended taxonomy 4→6 classes, LightGBM, Optuna tuning, model evaluation, documentation |
| **Zach** | Analyst & Project Manager | All EDA visualisations, group liaison, final report writing, presentations |
| **Htet Htet Wint** | Data Engineer | Spotify dataset sourcing, data loading, schema inspection, LR baseline |
| **Layaung Linn Lett** | Feature Engineer | Data cleaning, Winsorisation, genre taxonomy (4 classes), RF baseline |

---

## 🎯 Project Overview

Current music platforms recommend playlists by genre but do not adapt to the listener's **current mood**. MoodTunes AI solves this by:

1. Collapsing **114 Spotify genre labels → 6 mood super-genres** using iterative audio-based taxonomy
2. Engineering **42 features** from 15 raw Spotify audio features via a custom sklearn transformer
3. Training a **LightGBM classifier** achieving **69% accuracy** and **0.90 macro ROC-AUC**
4. Recommending the **top 5 most similar individual songs** — not playlists — using cosine similarity

---

## 📁 Project Structure

```
moodtunes-ai-group3/
├── EAAI_Group3_COMPLETE_FINAL.ipynb   ← Main Jupyter notebook (Google Colab)
├── app.py                              ← Streamlit web application
├── dataset.csv                         ← Spotify Tracks Dataset (114k tracks)
├── requirements.txt                    ← Python dependencies
├── .gitignore
├── README.md
├── models/
│   ├── final_model.pkl                 ← Trained LightGBM model
│   ├── label_encoder.pkl               ← Fitted LabelEncoder
│   ├── preprocessor.pkl                ← Fitted sklearn Pipeline
│   └── README.md
└── artifacts/
    ├── X_train.npy                     ← Preprocessed training arrays
    ├── X_val.npy
    ├── X_test.npy
    ├── y_train.npy
    ├── y_val.npy
    ├── y_test.npy
    └── README.md
```

---

## 🚀 How to Run

### Streamlit App (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/HeinHtet-Phyo/moodtunes-ai-group3.git
cd moodtunes-ai-group3

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Opens at: **http://localhost:8501**

### Jupyter Notebook (Google Colab)

1. Upload `EAAI_Group3_COMPLETE_FINAL.ipynb` to Google Colab
2. Upload `dataset.csv` to Google Drive at `My Drive/EAAI_Project/dataset.csv`
3. Run all cells — takes approximately 10–15 minutes

---

## ⚠️ How to Get the Model Files

The `models/` folder requires 3 `.pkl` files to run the Streamlit app.

**Option A — Google Colab (recommended)**

1. Run all cells in `EAAI_Group3_COMPLETE_FINAL.ipynb` on Google Colab
2. After Section 4 completes, go to `My Drive/EAAI_Project/models/`
3. Download and place these 3 files into the `models/` folder:
   - `final_model.pkl`
   - `label_encoder.pkl`
   - `preprocessor.pkl`

**Option B — Run locally**

```bash
pip install -r requirements.txt
jupyter notebook EAAI_Group3_COMPLETE_FINAL.ipynb
```

Run all cells — model files save to `models/` automatically.

---

## 📊 Results

| Metric | Value | Interpretation |
|---|---|---|
| **Macro ROC-AUC** | **0.90** | Excellent — near feature-space ceiling |
| Test Accuracy | 69% | Strong for 6-class audio classification |
| Weighted F1 | 0.68 | Gap vs ROC-AUC due to dance class imbalance (52%) |
| Macro F1 | 0.64 | Consistent across all 6 mood classes |

---

## 🎵 Recommender Modes

| Mode | Description |
|---|---|
| 🎭 By Mood | Pick happy / chill / focus / sad → top 5 songs |
| 🔍 By Song | Enter any song title → detect mood → top 5 similar |
| 🎛️ By Features | Adjust 8 audio sliders → predict mood → top 5 songs |
| 🎸 By Genre | Browse 6 super-genre categories → top 5 tracks |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| LightGBM | Primary classifier |
| Optuna | Hyperparameter tuning (TPE sampler) |
| scikit-learn | Pipeline, preprocessing, evaluation |
| Streamlit | Web application UI |
| pandas / numpy | Data processing |
| Spotify Tracks Dataset | 114,000 tracks, 21 features |

---

## 📚 References

- Pandya, M. (no date) *Spotify Tracks Dataset* [online]. Kaggle. Available at: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset [Accessed: 17 December 2025].
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T. (2017) 'LightGBM: A highly efficient gradient boosting decision tree', in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. New York: Curran Associates, pp. 3146–3154.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: A next-generation hyperparameter optimization framework', in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2019)*. New York: ACM, pp. 2623–2631.
- Salton, G. and McGill, M.J. (1983) *Introduction to Modern Information Retrieval*. New York: McGraw-Hill.
- Hand, D.J. and Till, R.J. (2001) 'A simple generalisation of the area under the ROC curve for multiple class problems', *Machine Learning*, 45(2), pp. 171–186.
- Russell, J.A. (1980) 'A circumplex model of affect', *Journal of Personality and Social Psychology*, 39(6), pp. 1161–1178.
- Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of deep bidirectional transformers for language understanding', in *Proceedings of NAACL-HLT 2019*. Minneapolis: Association for Computational Linguistics, pp. 4171–4186.
- Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. New York: Curran Associates, pp. 4765–4774.
- Ko, S. (2025) *HWNAS Dataset — Music Mood Classification* [online]. Kaggle. Available at: https://www.kaggle.com/datasets/stanislavko/hwnas-dataset-music-mood-classification [Accessed: 17 December 2025].

---

## 📋 Module Information

| Field | Detail |
|---|---|
| Module Code | UFCE3P-30-3 |
| Module Name | Essentials and Applications of Artificial Intelligence |
| Module Leaders | Dr Mahmoud Elbattah, Dr Sondess Missaoui |
| University | UWE Bristol |
| Submission | 21st April 2026 |
