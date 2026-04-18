# MoodTunes AI — Music Mood Classifier & Song Recommender

UFCE3P-30-3 — Essentials and Applications of Artificial Intelligence
Group 3 | UWE Bristol | April 2026

---

## Team

| Name | Role | Key Contributions |
|---|---|---|
| Hein Htet Phyo | AI Product Developer | Project concept, XGBoost baseline, top-5 recommender, Streamlit UI, GitHub, Report |
| Zulfiqar Khan | Lead ML Engineer | Extended taxonomy 4 to 6 classes, LightGBM, Optuna tuning, evaluation, documentation |
| Zach | Analyst & PM | EDA visualisations, group liaison, final report, presentations |
| Htet Htet Wint | Data Engineer | Spotify dataset sourcing, data loading, schema inspection, LR baseline |
| Layaung Linn Lett | Feature Engineer | Data cleaning, Winsorisation, genre taxonomy 4 classes, RF baseline |

---

## Project Overview

An end-to-end AI pipeline that:
1. Collapses 114 Spotify genre labels to 6 mood super-genres
2. Engineers 42 features from 15 raw audio features
3. Trains a LightGBM classifier (69% accuracy, 0.90 macro ROC-AUC)
4. Recommends top 5 individual songs by mood using cosine similarity

---

## Project Structure

```
EAAI_Project/
├── EAAI_Group3_COMPLETE_FINAL.ipynb   <- Main Jupyter notebook
├── app.py                              <- Streamlit web app
├── dataset.csv                         <- Spotify Tracks Dataset
├── requirements.txt
├── .gitignore
├── models/
│   ├── final_model.pkl
│   ├── label_encoder.pkl
│   ├── preprocessor.pkl
│   └── README.md
└── artifacts/
    ├── X_train.npy  X_val.npy  X_test.npy
    ├── y_train.npy  y_val.npy  y_test.npy
    └── README.md
```

---

## How to Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at: http://localhost:8501

---

## How to Get the Model Files

The models/ folder needs 3 pkl files. Generate them by running the notebook.

### Option A - Google Colab (recommended)

1. Open EAAI_Group3_COMPLETE_FINAL.ipynb in Google Colab
2. Upload dataset.csv to Google Drive at: My Drive/EAAI_Project/dataset.csv
3. Run all cells (about 10-15 minutes)
4. After Section 4 finishes, go to your Google Drive
5. Navigate to: My Drive/EAAI_Project/models/
6. Download these 3 files:
   - final_model.pkl
   - label_encoder.pkl
   - preprocessor.pkl
7. Place them in the models/ folder of this project

### Option B - Run locally

```bash
pip install -r requirements.txt
jupyter notebook EAAI_Group3_COMPLETE_FINAL.ipynb
```

Run all cells. Model files save to models/ automatically.

---

## Results

| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.90 |
| Test Accuracy | 69% |
| Weighted F1 | 0.68 |
| Macro F1 | 0.64 |

---

## Recommender Modes

- By Mood: happy / chill / focus / sad
- By Song Title: enter any song name
- By Audio Features: 8 audio sliders
- By Genre: 6 super-genre categories

---

## References

- Pandya, M. (no date) Spotify Tracks Dataset. Kaggle.
- Ke, G. et al. (2017) LightGBM. NeurIPS 30, pp. 3146-3154.
- Akiba, T. et al. (2019) Optuna. KDD, pp. 2623-2631.
- Salton, G. and McGill, M.J. (1983) Introduction to Modern Information Retrieval.
- Russell, J.A. (1980) A circumplex model of affect. JPSP 39(6), pp. 1161-1178.

---

Module: UFCE3P-30-3 | UWE Bristol | Submission: 21 April 2026
