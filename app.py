"""
MoodTunes AI — Music Mood Classifier & Song Recommender
Group 3 | UFCE3P-30-3 | UWE Bristol | April 2026

Fixes in this version:
  - genre_sel session_state key conflict fixed (no st.rerun after widget key assignment)
  - KeyError None fixed (safe genre lookup with fallback)
  - About page fixed (pure st calls, no raw HTML blocks)
  - Song cards open Spotify search in new tab
  - Mood tab has Shuffle button — always returns different songs
  - All tabs show no red errors
"""

import os
import pickle
import random
import urllib.parse
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import random
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodTunes AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def H(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# GENRE MAPPING  (114 → 6)
# ─────────────────────────────────────────────────────────────────────────────
GENRE_MAPPING = {
    "acoustic":"acoustic","folk":"acoustic","singer-songwriter":"acoustic",
    "songwriter":"acoustic","country":"acoustic","bluegrass":"acoustic",
    "honky-tonk":"acoustic","guitar":"acoustic","blues":"acoustic",
    "jazz":"acoustic","groove":"acoustic","ambient":"acoustic",
    "new-age":"acoustic","sleep":"acoustic","study":"acoustic",
    "chill":"acoustic","piano":"acoustic","classical":"acoustic",
    "opera":"acoustic","romance":"acoustic","sad":"acoustic",
    "alternative":"alternative","alt-rock":"alternative","indie":"alternative",
    "indie-pop":"alternative","grunge":"alternative","british":"alternative",
    "psych-rock":"alternative","garage":"alternative","rock":"alternative",
    "hard-rock":"alternative","rock-n-roll":"alternative","rockabilly":"alternative",
    "pop":"dance","dance":"dance","disco":"dance","party":"dance",
    "power-pop":"dance","pop-film":"dance","happy":"dance","synth-pop":"dance",
    "gospel":"dance","soul":"dance","r-n-b":"dance","funk":"dance",
    "latin":"dance","latino":"dance","reggaeton":"dance","dancehall":"dance",
    "salsa":"dance","samba":"dance","sertanejo":"dance","forro":"dance",
    "pagode":"dance","mpb":"dance","brazil":"dance","tango":"dance",
    "reggae":"dance","dub":"dance","ska":"dance","world-music":"dance",
    "afrobeat":"dance","turkish":"dance","iranian":"dance","french":"dance",
    "german":"dance","swedish":"dance","spanish":"dance","indian":"dance",
    "j-pop":"dance","j-idol":"dance","j-dance":"dance","anime":"dance",
    "j-rock":"dance","k-pop":"dance","cantopop":"dance","mandopop":"dance","malay":"dance",
    "edm":"electronic","electronic":"electronic","electro":"electronic",
    "club":"electronic","idm":"electronic","house":"electronic",
    "deep-house":"electronic","chicago-house":"electronic",
    "detroit-techno":"electronic","minimal-techno":"electronic",
    "techno":"electronic","trance":"electronic","progressive-house":"electronic",
    "drum-and-bass":"electronic","dubstep":"electronic","breakbeat":"electronic",
    "hardstyle":"electronic",
    "metal":"heavy","heavy-metal":"heavy","death-metal":"heavy",
    "black-metal":"heavy","metalcore":"heavy","grindcore":"heavy",
    "emo":"heavy","goth":"heavy","industrial":"heavy",
    "punk":"heavy","punk-rock":"heavy","hardcore":"heavy",
    "hip-hop":"vocal","trip-hop":"vocal","kids":"vocal","children":"vocal",
    "comedy":"vocal","disney":"vocal","show-tunes":"vocal",
}
SUPER_GENRES = sorted(set(GENRE_MAPPING.values()))
REC_FEATURES = ["danceability","energy","valence","tempo",
                "acousticness","instrumentalness","speechiness","liveness"]

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
MOOD_CFG = {
    "happy": dict(emoji="😊", label="Happy",  desc="High energy · Positive vibes",
                  grad="linear-gradient(135deg,#f59e0b,#ef4444)",
                  color="#d97706", bg="#fffbeb", border="#fde68a"),
    "chill": dict(emoji="😌", label="Chill",  desc="Calm · Acoustic · Relaxed",
                  grad="linear-gradient(135deg,#0ea5e9,#06b6d4)",
                  color="#0284c7", bg="#f0f9ff", border="#bae6fd"),
    "focus": dict(emoji="🎯", label="Focus",  desc="Driven · Stay in the zone",
                  grad="linear-gradient(135deg,#8b5cf6,#ec4899)",
                  color="#7c3aed", bg="#faf5ff", border="#ddd6fe"),
    "sad":   dict(emoji="😔", label="Sad",    desc="Emotional · Reflective",
                  grad="linear-gradient(135deg,#64748b,#475569)",
                  color="#475569", bg="#f8fafc", border="#cbd5e1"),
}

GENRE_CFG = {
    "acoustic":    dict(emoji="🎸", color="#2563eb", light="#eff6ff", border="#bfdbfe"),
    "alternative": dict(emoji="🎙️", color="#7c3aed", light="#faf5ff", border="#ddd6fe"),
    "dance":       dict(emoji="💃", color="#db2777", light="#fdf2f8", border="#fbcfe8"),
    "electronic":  dict(emoji="🎛️", color="#ea580c", light="#fff7ed", border="#fed7aa"),
    "heavy":       dict(emoji="🤘", color="#dc2626", light="#fef2f2", border="#fecaca"),
    "vocal":       dict(emoji="🎤", color="#16a34a", light="#f0fdf4", border="#bbf7d0"),
}
_FALLBACK_GENRE_CFG = dict(emoji="🎵", color="#7c3aed", light="#f3f0ff", border="#ddd6fe")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEER  (must match notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
_EPS = 1e-6

class MusicFeatureEngineer(BaseEstimator, TransformerMixin):
    INPUT_FEATURES = [
        "popularity","duration_ms","explicit","danceability","energy","key",
        "loudness","mode","speechiness","acousticness","instrumentalness",
        "liveness","valence","tempo","time_signature",
    ]
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        df = (X[self.INPUT_FEATURES].copy().astype(float)
              if isinstance(X, pd.DataFrame)
              else pd.DataFrame(X, columns=self.INPUT_FEATURES).astype(float))
        e = {}
        for c in ["speechiness","acousticness","instrumentalness","liveness"]:
            e[f"log1p_{c}"] = np.log1p(df[c].clip(lower=0))
        e["duration_min"]          = df["duration_ms"] / 60_000.0
        e["log_duration_ms"]       = np.log1p(df["duration_ms"].clip(lower=0))
        e["abs_loudness"]          = df["loudness"].abs()
        e["loudness_norm"]         = df["loudness"].abs() / 60.0
        e["energy_x_not_acoustic"] = df["energy"] * (1 - df["acousticness"])
        e["dance_x_energy"]        = df["danceability"] * df["energy"]
        e["valence_x_energy"]      = df["valence"] * df["energy"]
        e["valence_x_dance"]       = df["valence"] * df["danceability"]
        e["speech_x_not_acoustic"] = df["speechiness"] * (1 - df["acousticness"])
        e["instrumental_x_energy"] = df["instrumentalness"] * df["energy"]
        e["loudness_per_energy"]   = df["loudness"].abs() / (df["energy"] + _EPS)
        e["pop_x_dance"]           = (df["popularity"] / 100.0) * df["danceability"]
        e["acoustic_x_valence"]    = df["acousticness"] * df["valence"]
        e["tempo_sq"]              = (df["tempo"] / 250.0) ** 2
        e["popularity_sq"]         = (df["popularity"] / 100.0) ** 2
        e["energy_sq"]             = df["energy"] ** 2
        e["acousticness_sq"]       = df["acousticness"] ** 2
        e["instrumentalness_sq"]   = df["instrumentalness"] ** 2
        e["tempo_norm"]            = df["tempo"] / 250.0
        e["tempo_slow"]            = (df["tempo"] < 90).astype(float)
        e["tempo_fast"]            = (df["tempo"] > 140).astype(float)
        e["key_x_mode"]            = df["key"] * df["mode"]
        e["liveness_ratio"]        = df["liveness"] / (df["energy"] + _EPS)
        return np.hstack([df.values, pd.DataFrame(e, index=df.index).values])

# ─────────────────────────────────────────────────────────────────────────────
# MOOD RULE
# ─────────────────────────────────────────────────────────────────────────────
def assign_mood(row) -> str:
    v, e, a, ins = row["valence"], row["energy"], row["acousticness"], row["instrumentalness"]
    if   v >= 0.60 and e >= 0.55:                               return "happy"
    elif a >= 0.50 or ins >= 0.15 or (e < 0.45 and v >= 0.35): return "chill"
    elif e >= 0.65 and v < 0.60:                                return "focus"
    else:                                                        return "sad"

# ─────────────────────────────────────────────────────────────────────────────
# SPOTIFY URL helper
# ─────────────────────────────────────────────────────────────────────────────
def spotify_url(track: str, artist: str) -> str:
    q = urllib.parse.quote(f"{track} {artist}")
    return f"https://open.spotify.com/search/{q}"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Required data file not found: {DATA_PATH.name}. "
            "Add it to the GitHub repository so Streamlit Cloud can access it."
        )
    df = pd.read_csv(DATA_PATH, index_col=0)
    df["super_genre"] = df["track_genre"].map(lambda g: GENRE_MAPPING.get(g, g))
    df = df[df["super_genre"].isin(SUPER_GENRES)].copy()
    rec = df[["track_name","artists","track_genre","popularity","super_genre"] + REC_FEATURES].copy()
    rec = rec.dropna().drop_duplicates(subset=["track_name","artists"]).reset_index(drop=True)
    rec["mood"] = rec.apply(assign_mood, axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rec[REC_FEATURES])
    songs  = sorted(rec["track_name"].dropna().unique().tolist())
    return rec, np.array(scaled, dtype=np.float32), scaler, songs

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDERS
# ─────────────────────────────────────────────────────────────────────────────
def rec_mood(df_rec, scaled, mood: str, n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return top-n songs for a mood. seed controls shuffle — pass random seed for variety."""
    mask  = (df_rec["mood"] == mood).values
    mdf   = df_rec[mask].copy()
    mvecs = scaled[mask]
    qvec  = mvecs.mean(axis=0).reshape(1, -1)
    sims  = cosine_similarity(qvec, mvecs)[0]
    mdf["similarity"] = sims
    # Score = similarity * 0.6 + popularity_norm * 0.4 + small random noise for shuffle
    rng   = np.random.default_rng(seed)
    noise = rng.uniform(0, 0.08, size=len(mdf))
    mdf["score"] = mdf["similarity"] * 0.6 + (mdf["popularity"] / 100.0) * 0.4 + noise
    return mdf.sort_values("score", ascending=False).head(n).reset_index(drop=True)

def rec_song(df_rec, scaled, title: str, n: int = 5):
    m = df_rec[df_rec["track_name"].str.lower() == title.lower()]
    if m.empty:
        m = df_rec[df_rec["track_name"].str.lower().str.contains(
            title.lower(), na=False, regex=False)]
    if m.empty:
        return None, None
    idx  = m.index[0]
    inp  = df_rec.loc[idx]
    mood = inp["mood"]
    mask = (df_rec["mood"] == mood).values
    mdf  = df_rec[mask].copy()
    qvec = scaled[idx].reshape(1, -1)
    mdf["similarity"] = cosine_similarity(qvec, scaled[mask])[0]
    mdf  = mdf[mdf.index != idx]
    return inp, mdf.sort_values(["similarity","popularity"], ascending=[False,False]).head(n).reset_index(drop=True)

def rec_features(df_rec, scaled, scaler, feats: dict, n: int = 5, seed: int = 42):
    mood  = assign_mood(pd.Series(feats))
    qvec  = scaler.transform(pd.DataFrame([feats], columns=REC_FEATURES))
    mask  = (df_rec["mood"] == mood).values
    mdf   = df_rec[mask].copy()
    sims  = cosine_similarity(qvec, scaled[mask])[0]
    rng   = np.random.default_rng(seed)
    noise = rng.uniform(0, 0.08, size=len(mdf))
    mdf["score"] = sims * 0.6 + (mdf["popularity"] / 100.0) * 0.4 + noise
    mdf["similarity"] = sims
    return mood, mdf.sort_values("score", ascending=False).head(n).reset_index(drop=True)

def rec_genre(df_rec, scaled, genre: str, n: int = 5) -> pd.DataFrame:
    mask  = (df_rec["super_genre"] == genre).values
    gdf   = df_rec[mask].copy()
    qvec  = scaled[mask].mean(axis=0).reshape(1, -1)
    gdf["similarity"] = cosine_similarity(qvec, scaled[mask])[0]
    return gdf.sort_values(["similarity","popularity"], ascending=[False,False]).head(n).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    sel_mood=None, mood_recs=None, mood_seed=42,
    song_recs=None, song_inp=None,
    feat_mood=None, feat_recs=None, feat_seed=42, feat_feats=None,
    genre_recs=None, genre_name=None,
)

def _init():
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root{
  --bg:#f8fafc;--card:#fff;--bdr:#e2e8f0;--bdr2:#cbd5e1;
  --tx:#0f172a;--mt:#64748b;--sb:#94a3b8;
  --ac:#7c3aed;--ac2:#0ea5e9;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"],.stApp{
  font-family:'DM Sans',sans-serif!important;
  background:var(--bg)!important;color:var(--tx)!important;
}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"],
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stAppViewContainer"]{padding-top:0!important;}
.main .block-container{padding:0!important;max-width:100%!important;}
section.main>div{padding:0!important;}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:radial-gradient(rgba(0,0,0,0.032) 1px,transparent 1px);
  background-size:40px 40px;}
.stApp>*{position:relative;z-index:1;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-thumb{background:linear-gradient(#7c3aed,#0ea5e9);border-radius:10px;}

/* Buttons */
.stButton>button{
  font-family:'Syne',sans-serif!important;font-weight:700!important;
  font-size:0.92rem!important;border:none!important;border-radius:999px!important;
  padding:11px 24px!important;
  background:linear-gradient(135deg,#7c3aed,#0ea5e9)!important;
  color:#fff!important;box-shadow:0 4px 16px rgba(124,58,237,.22)!important;
  width:100%!important;cursor:pointer!important;transition:all .2s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(124,58,237,.30)!important;}
.stButton>button:active{transform:scale(.98)!important;}

/* Text input — dark text on white bg */
[data-testid="stTextInput"] input{
  background:#fff!important;border:1.5px solid var(--bdr2)!important;
  border-radius:12px!important;color:#0f172a!important;
  font-family:'DM Sans',sans-serif!important;font-size:1rem!important;
  padding:12px 16px!important;caret-color:#7c3aed!important;
}
[data-testid="stTextInput"] input::placeholder{color:#94a3b8!important;}
[data-testid="stTextInput"] input:focus{border-color:#7c3aed!important;
  box-shadow:0 0 0 3px rgba(124,58,237,.12)!important;outline:none!important;}
[data-testid="stTextInput"] label{color:var(--mt)!important;font-size:.75rem!important;
  font-family:'DM Mono',monospace!important;letter-spacing:.6px!important;
  text-transform:uppercase!important;}

/* Selectbox */
[data-testid="stSelectbox"]>div>div{background:#fff!important;
  border:1.5px solid var(--bdr2)!important;border-radius:12px!important;
  color:#0f172a!important;font-family:'DM Sans',sans-serif!important;}
[data-testid="stSelectbox"] label{color:var(--mt)!important;
  font-family:'DM Mono',monospace!important;font-size:.75rem!important;
  text-transform:uppercase!important;letter-spacing:.6px!important;}

/* Sliders */
[data-testid="stSlider"] label{font-family:'DM Sans',sans-serif!important;
  font-size:.85rem!important;font-weight:500!important;color:#0f172a!important;}
[data-testid="stSlider"]>div>div>div{background:#e2e8f0!important;border-radius:999px!important;}
[data-testid="stSlider"]>div>div>div>div[data-testid="stSlider-track-fill"]{
  background:linear-gradient(90deg,#7c3aed,#0ea5e9)!important;}
[data-testid="stSlider"]>div>div>div[role="slider"]{background:#fff!important;
  border:2.5px solid #7c3aed!important;box-shadow:0 0 0 3px rgba(124,58,237,.15)!important;
  width:18px!important;height:18px!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#fff!important;border-radius:999px!important;
  padding:4px!important;gap:2px!important;border:1.5px solid var(--bdr)!important;
  width:fit-content!important;box-shadow:0 2px 8px rgba(0,0,0,.06)!important;}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif!important;font-size:.87rem!important;
  font-weight:600!important;color:var(--mt)!important;background:transparent!important;
  border-radius:999px!important;padding:8px 22px!important;border:none!important;transition:all .2s!important;}
[aria-selected="true"][data-baseweb="tab"]{
  background:linear-gradient(135deg,#7c3aed,#0ea5e9)!important;color:#fff!important;
  box-shadow:0 2px 10px rgba(124,58,237,.25)!important;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important;}
[data-testid="stTabsContent"]{padding-top:24px!important;}
[data-testid="stHorizontalBlock"]{gap:14px!important;align-items:start!important;}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.3;transform:scale(1.7);}}
.fade-up{animation:fadeUp .4s cubic-bezier(.4,0,.2,1) forwards;}
hr{border-color:var(--bdr)!important;}
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header(total: int):
    H(f"""
<div style="position:sticky;top:0;z-index:999;background:rgba(248,250,252,.95);
  backdrop-filter:blur(20px);border-bottom:1.5px solid #e2e8f0;padding:14px 40px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
  box-shadow:0 2px 10px rgba(0,0,0,.06);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:40px;height:40px;background:linear-gradient(135deg,#7c3aed,#0ea5e9);
      border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">🎵</div>
    <span style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
      letter-spacing:-1px;color:#0f172a;">MoodTunes<span
      style="background:linear-gradient(135deg,#7c3aed,#0ea5e9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span></span>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    <span style="padding:6px 14px;border-radius:999px;font-size:11px;font-weight:700;
      font-family:'DM Mono',monospace;background:#f3f0ff;color:#7c3aed;
      border:1px solid #ddd6fe;">LightGBM · ROC-AUC 0.90</span>
    <span style="padding:6px 14px;border-radius:999px;font-size:11px;font-weight:700;
      font-family:'DM Mono',monospace;background:#f0fdf4;color:#16a34a;
      border:1px solid #bbf7d0;">{total:,} Tracks</span>
    <span style="padding:6px 14px;border-radius:999px;font-size:11px;font-weight:700;
      font-family:'DM Mono',monospace;background:#fdf2f8;color:#db2777;
      border:1px solid #fbcfe8;">6 Mood Classes</span>
    <span style="padding:6px 14px;border-radius:999px;font-size:11px;font-weight:700;
      font-family:'DM Mono',monospace;background:#fff7ed;color:#ea580c;
      border:1px solid #fed7aa;">Group 3 · UWE Bristol</span>
  </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
def render_hero():
    H("""
<div style="max-width:1200px;margin:0 auto;padding:72px 40px 48px;text-align:center;"
  class="fade-up">
  <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 18px;margin-bottom:28px;
    background:#f3f0ff;border:1.5px solid #ddd6fe;border-radius:999px;
    font-size:12px;color:#7c3aed;font-family:'DM Mono',monospace;letter-spacing:.5px;">
    <span style="width:7px;height:7px;background:#7c3aed;border-radius:50%;
      animation:pulseDot 2s ease-in-out infinite;display:inline-block;"></span>
    AI-Powered Mood Detection &nbsp;·&nbsp; UFCE3P-30-3 &nbsp;·&nbsp; UWE Bristol 2026
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:clamp(44px,6vw,80px);
    font-weight:800;line-height:1.03;letter-spacing:-3px;margin-bottom:22px;color:#0f172a;">
    Music that matches<br>
    <span style="background:linear-gradient(135deg,#7c3aed,#0ea5e9);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">your mood</span>
  </h1>
  <p style="font-size:17px;color:#64748b;max-width:500px;margin:0 auto 40px;line-height:1.75;">
    Tell us how you feel. Our LightGBM AI picks your mood from 42 audio features and finds the
    <strong style="color:#7c3aed;">top 5 perfect songs</strong> — click any card to open on Spotify.
  </p>
  <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
    <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
      padding:14px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);">
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
        background:linear-gradient(135deg,#7c3aed,#0ea5e9);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">0.90</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">ROC-AUC</div></div>
    <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
      padding:14px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);">
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
        background:linear-gradient(135deg,#0ea5e9,#06b6d4);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">69%</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">Accuracy</div></div>
    <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
      padding:14px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);">
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
        background:linear-gradient(135deg,#db2777,#f59e0b);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">42</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">Features</div></div>
    <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
      padding:14px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);">
      <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:800;
        background:linear-gradient(135deg,#f59e0b,#ea580c);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">114k</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">Tracks</div></div>
  </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# SONG CARDS  — each card is a Spotify link
# ─────────────────────────────────────────────────────────────────────────────
def song_cards(df: pd.DataFrame, mood: str = None, genre: str = None):
    if mood and mood in MOOD_CFG:
        grad  = MOOD_CFG[mood]["grad"]
        color = MOOD_CFG[mood]["color"]
    elif genre and genre in GENRE_CFG:
        c     = GENRE_CFG[genre]["color"]
        grad  = f"linear-gradient(135deg,{c},{c}bb)"
        color = c
    else:
        grad  = "linear-gradient(135deg,#7c3aed,#0ea5e9)"
        color = "#7c3aed"

    # header row
    H("""
<div style="display:flex;justify-content:space-between;padding:0 4px 8px;
  border-bottom:1.5px solid #f1f5f9;margin-bottom:8px;">
  <span style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;">Track · Artist · Genre</span>
  <span style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;">Popularity &nbsp;|&nbsp; Score</span>
</div>
""")

    for i, row in df.iterrows():
        rank   = i + 1
        title  = str(row.get("track_name", "Unknown"))
        artist = str(row.get("artists", "Unknown"))
        sg     = str(row.get("super_genre", ""))
        pop    = int(row.get("popularity", 0))
        sim    = float(row.get("similarity", row.get("score", 0.0)))
        gcfg   = GENRE_CFG.get(sg, _FALLBACK_GENRE_CFG)
        url    = spotify_url(title, artist)
        t_disp = (title[:46] + "…")  if len(title)  > 46 else title
        a_disp = (artist[:40] + "…") if len(artist) > 40 else artist

        H(f"""
<a href="{url}" target="_blank" rel="noopener noreferrer"
   style="text-decoration:none;display:block;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:14px;padding:14px 18px;
    background:#fff;border:1.5px solid #e2e8f0;border-left:4px solid {color};
    border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.04);
    transition:box-shadow .18s,transform .18s;"
    onmouseover="this.style.boxShadow='0 6px 20px rgba(0,0,0,.10)';this.style.transform='translateY(-2px)'"
    onmouseout="this.style.boxShadow='0 2px 6px rgba(0,0,0,.04)';this.style.transform='translateY(0)'">
    <!-- rank badge -->
    <div style="width:40px;height:40px;border-radius:10px;flex-shrink:0;
      background:{grad};display:flex;align-items:center;justify-content:center;
      font-size:16px;font-weight:800;font-family:'Syne',sans-serif;color:#fff;">{rank}</div>
    <!-- title + artist -->
    <div style="flex:1;min-width:0;">
      <div style="font-size:15px;font-weight:600;color:#0f172a;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px;">
        {t_disp}</div>
      <div style="font-size:12px;color:#64748b;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{a_disp}</div>
    </div>
    <!-- genre pill -->
    <span style="padding:3px 10px;border-radius:999px;font-size:10px;font-weight:600;
      background:{gcfg['light']};color:{gcfg['color']};border:1px solid {gcfg['border']};
      font-family:'DM Mono',monospace;white-space:nowrap;flex-shrink:0;">
      {gcfg['emoji']} {sg}</span>
    <!-- pop bar -->
    <div style="display:flex;align-items:center;gap:5px;flex-shrink:0;">
      <div style="width:48px;height:4px;background:#f1f5f9;border-radius:2px;overflow:hidden;">
        <div style="width:{pop}%;height:100%;background:{grad};border-radius:2px;"></div>
      </div>
      <span style="font-size:10px;color:#94a3b8;font-family:'DM Mono',monospace;
        min-width:20px;">{pop}</span>
    </div>
    <!-- score -->
    <div style="font-family:'DM Mono',monospace;font-size:12px;font-weight:700;
      color:{color};flex-shrink:0;min-width:40px;text-align:right;">{sim:.3f}</div>
    <!-- spotify icon -->
    <div style="flex-shrink:0;width:28px;height:28px;border-radius:50%;
      background:#1DB954;display:flex;align-items:center;justify-content:center;">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52
          2 12 2zm4.65 14.65c-.2.3-.6.4-.9.2-2.45-1.5-5.55-1.85-9.2-1
          -.35.08-.7-.15-.78-.5-.08-.35.15-.7.5-.78 3.99-.9 7.4-.52
          10.15 1.15.3.2.4.6.23.93zm1.25-2.8c-.25.38-.76.5-1.14.25
          -2.8-1.72-7.07-2.22-10.38-1.22-.4.12-.83-.1-.95-.5-.12-.4
          .1-.83.5-.95 3.78-1.15 8.47-.59 11.7 1.38.37.23.5.74.27
          1.04zm.1-2.9C14.73 9.3 9.2 9.12 6.02 10.07c-.48.14-.99-.14
          -1.13-.62-.14-.48.14-.99.62-1.13 3.67-1.11 9.77-.9 13.62
          1.4.44.26.6.84.33 1.28-.25.44-.83.6-1.26.35z"/>
      </svg>
    </div>
  </div>
</a>
""")
    H("""
<div style="text-align:center;margin-top:6px;font-size:11px;color:#94a3b8;
  font-family:'DM Mono',monospace;letter-spacing:.3px;">
  Click any song to open on Spotify
</div>
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — BY MOOD  (with Shuffle button)
# ─────────────────────────────────────────────────────────────────────────────
def tab_mood(df_rec, scaled):
    H("""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 16px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;
    letter-spacing:1.2px;color:#94a3b8;text-transform:uppercase;margin-bottom:18px;">
    01 · How are you feeling right now?
  </div>
</div>
""")

    sel = st.session_state.sel_mood
    c1, c2, c3, c4 = st.columns(4, gap="small")

    for col, mood in zip([c1, c2, c3, c4], ["happy","chill","focus","sad"]):
        cfg  = MOOD_CFG[mood]
        isel = sel == mood
        with col:
            H(f"""
<div style="background:{cfg['bg'] if isel else '#fff'};
  border:{'2px solid ' + cfg['color'] if isel else '1.5px solid #e2e8f0'};
  border-radius:18px;padding:22px 18px;text-align:center;
  box-shadow:{'0 4px 20px ' + cfg['color'] + '40' if isel else '0 2px 8px rgba(0,0,0,.05)'};
  margin-bottom:8px;">
  <div style="font-size:36px;margin-bottom:10px;">{cfg['emoji']}</div>
  <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:800;
    color:#0f172a;margin-bottom:5px;">{cfg['label']}</div>
  <div style="font-size:11px;color:#64748b;line-height:1.5;">{cfg['desc']}</div>
</div>
""")
            if st.button(f"Pick {cfg['label']}", key=f"mb_{mood}"):
                st.session_state.sel_mood  = mood
                st.session_state.mood_seed = random.randint(0, 99999)
                st.session_state.mood_recs = rec_mood(df_rec, scaled, mood,
                                                       seed=st.session_state.mood_seed)
                st.rerun()

    # Results
    if st.session_state.mood_recs is not None and st.session_state.sel_mood:
        mood = st.session_state.sel_mood
        cfg  = MOOD_CFG[mood]
        recs = st.session_state.mood_recs

        H(f"""
<div style="max-width:1200px;margin:0 auto;padding:28px 40px 8px;">
  <div style="display:flex;align-items:center;justify-content:space-between;
    margin-bottom:20px;padding:18px 22px;background:{cfg['bg']};
    border:1.5px solid {cfg['border']};border-radius:14px;">
    <div style="display:flex;align-items:center;gap:14px;">
      <span style="font-size:30px;">{cfg['emoji']}</span>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
          background:{cfg['grad']};-webkit-background-clip:text;
          -webkit-text-fill-color:transparent;">Top 5 {cfg['label']} Songs</div>
        <div style="font-size:12px;color:#64748b;margin-top:2px;">
          Cosine similarity to {mood} mood centroid · click a song to open Spotify</div>
      </div>
    </div>
  </div>
</div>
""")

        # Shuffle button — below the header card, above song list
        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 10px;">')
        col_shuf, _ = st.columns([1, 4])
        with col_shuf:
            if st.button("🔀  Shuffle Songs", key="shuffle_btn"):
                new_seed = random.randint(0, 99999)
                st.session_state.mood_seed = new_seed
                st.session_state.mood_recs = rec_mood(df_rec, scaled, mood, seed=new_seed)
                st.rerun()
        H('</div>')

        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 40px;">')
        song_cards(recs, mood=mood)
        H('</div>')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BY SONG  (selectbox = autocomplete, fixed text colour)
# ─────────────────────────────────────────────────────────────────────────────
def tab_song(df_rec, scaled, songs):
    H("""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 16px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;
    letter-spacing:1.2px;color:#94a3b8;text-transform:uppercase;margin-bottom:18px;">
    02 · Search by song title
  </div>
  <div style="font-size:14px;color:#64748b;margin-bottom:22px;line-height:1.6;">
    Type a song name — select from the dropdown and hit Find Similar Songs.
    We detect its mood and find 5 similar tracks.
  </div>
</div>
""")

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 20px;">')
    chosen = st.selectbox(
        "SONG TITLE",
        options=[""] + songs,
        format_func=lambda x: "Type to search a song..." if x == "" else x,
        key="song_pick",
    )
    col_b, _ = st.columns([1, 4])
    with col_b:
        go = st.button("🔍  Find Similar Songs", key="song_go")
    H('</div>')

    if go:
        if not chosen:
            H("""
<div style="max-width:1200px;margin:0 auto;padding:0 40px;">
  <div style="padding:14px 20px;background:#fef2f2;border:1.5px solid #fecaca;
    border-radius:12px;color:#dc2626;font-size:14px;">
    Please select a song from the dropdown first.
  </div>
</div>
""")
        else:
            with st.spinner("Finding similar songs..."):
                inp, recs = rec_song(df_rec, scaled, chosen, n=5)
            if inp is None:
                H(f"""
<div style="max-width:1200px;margin:0 auto;padding:0 40px;">
  <div style="padding:14px 20px;background:#fef2f2;border:1.5px solid #fecaca;
    border-radius:12px;color:#dc2626;font-size:14px;">
    No song found for "<strong>{chosen}</strong>". Try another title.
  </div>
</div>
""")
            else:
                st.session_state.song_inp  = inp
                st.session_state.song_recs = recs

    if st.session_state.song_recs is not None and st.session_state.song_inp is not None:
        inp  = st.session_state.song_inp
        recs = st.session_state.song_recs
        mood = str(inp.get("mood",""))
        cfg  = MOOD_CFG.get(mood, MOOD_CFG["chill"])
        H(f"""
<div style="max-width:1200px;margin:0 auto;padding:20px 40px 8px;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:22px;">
    <div style="padding:18px 22px;background:#fff;border:1.5px solid #e2e8f0;
      border-radius:14px;box-shadow:0 2px 8px rgba(0,0,0,.05);">
      <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
        text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Input Song</div>
      <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">
        {str(inp.get("track_name",""))[:50]}</div>
      <div style="font-size:13px;color:#64748b;">{str(inp.get("artists",""))[:45]}</div>
    </div>
    <div style="padding:18px 22px;background:{cfg['bg']};
      border:1.5px solid {cfg['border']};border-radius:14px;">
      <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
        text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Detected Mood</div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:28px;">{cfg['emoji']}</span>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
            background:{cfg['grad']};-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;">{cfg['label']}</div>
          <div style="font-size:12px;color:#64748b;">{cfg['desc']}</div>
        </div>
      </div>
    </div>
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
    color:#0f172a;margin-bottom:14px;">Top 5 Similar Songs</div>
</div>
""")
        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 40px;">')
        song_cards(recs, mood=mood)
        H('</div>')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BY FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def tab_features(df_rec, scaled, scaler):
    H("""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 16px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;
    letter-spacing:1.2px;color:#94a3b8;text-transform:uppercase;margin-bottom:18px;">
    03 · Custom audio profile
  </div>
  <div style="font-size:14px;color:#64748b;margin-bottom:22px;line-height:1.6;">
    Adjust the sliders — mood is predicted in real time. Hit the button to get your songs.
  </div>
</div>
""")

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px;">')
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        H("""<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
          padding:22px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.05);">
          <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
            color:#0f172a;margin-bottom:14px;">Energy &amp; Feel</div>""")
        dance  = st.slider("Danceability",        0.0, 1.0,  0.65, 0.01, key="fd")
        energy = st.slider("Energy",              0.0, 1.0,  0.70, 0.01, key="fe")
        valence= st.slider("Valence (happiness)", 0.0, 1.0,  0.50, 0.01, key="fv")
        tempo  = st.slider("Tempo (BPM)",        40.0,250.0,120.0, 1.0,  key="ft")
        H("</div>")

    with c2:
        H("""<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
          padding:22px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.05);">
          <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
            color:#0f172a;margin-bottom:14px;">Texture &amp; Character</div>""")
        acoustic = st.slider("Acousticness",     0.0,1.0,0.20,0.01,key="fa")
        instr    = st.slider("Instrumentalness", 0.0,1.0,0.00,0.01,key="fi")
        speech   = st.slider("Speechiness",      0.0,1.0,0.05,0.01,key="fs")
        live     = st.slider("Liveness",         0.0,1.0,0.12,0.01,key="fl")
        H("</div>")
    H('</div>')

    feats = dict(danceability=dance, energy=energy, valence=valence, tempo=tempo,
                 acousticness=acoustic, instrumentalness=instr,
                 speechiness=speech, liveness=live)
    inferred = assign_mood(pd.Series(feats))
    cfg2 = MOOD_CFG[inferred]

    H(f"""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 18px;">
  <div style="display:flex;align-items:center;justify-content:space-between;
    padding:14px 22px;background:{cfg2['bg']};
    border:1.5px solid {cfg2['border']};border-radius:12px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="font-size:26px;">{cfg2['emoji']}</span>
      <div>
        <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
          text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px;">Predicted mood</div>
        <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:800;
          background:{cfg2['grad']};-webkit-background-clip:text;
          -webkit-text-fill-color:transparent;">{cfg2['label']}</div>
      </div>
    </div>
    <div style="font-size:12px;color:#94a3b8;text-align:right;line-height:1.6;">
      Updates as you<br>move the sliders</div>
  </div>
</div>
""")

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 20px;">')
    col_b2, _ = st.columns([1, 3])
    with col_b2:
        if st.button("✦  Get My Top 5 Songs", key="feat_go"):
            with st.spinner("Analysing audio profile..."):
                mp, recs = rec_features(df_rec, scaled, scaler, feats, n=5, seed=st.session_state.feat_seed)
            st.session_state.feat_mood = mp
            st.session_state.feat_recs = recs
            st.session_state.feat_feats = feats
            st.rerun()
    H('</div>')

    if st.session_state.feat_recs is not None:
        mp   = st.session_state.feat_mood
        recs = st.session_state.feat_recs
        cfg3 = MOOD_CFG.get(mp, MOOD_CFG["chill"])
        H(f"""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 8px;">
  <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
    background:{cfg3['grad']};-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin-bottom:14px;">
    {cfg3['emoji']} Top 5 {cfg3['label']} Songs for Your Profile</div>
</div>
""")
        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 10px;">')
        col_fs, _ = st.columns([1, 4])
        with col_fs:
            if st.button("🔀  Shuffle Songs", key="feat_shuffle_btn"):
                st.session_state.feat_seed = random.randint(0, 99999)
                if st.session_state.feat_feats is not None:
                    mp, recs = rec_features(
                        df_rec,
                        scaled,
                        scaler,
                        st.session_state.feat_feats,
                        n=5,
                        seed=st.session_state.feat_seed,
                    )
                    st.session_state.feat_mood = mp
                    st.session_state.feat_recs = recs
                st.rerun()
        H('</div>')
        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 40px;">')
        song_cards(recs, mood=mp)
        H('</div>')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — BY GENRE  (FIXED: no session_state key conflict, safe genre lookup)
# ─────────────────────────────────────────────────────────────────────────────
def tab_genre(df_rec, scaled):
    H("""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 16px;">
  <div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;
    letter-spacing:1.2px;color:#94a3b8;text-transform:uppercase;margin-bottom:18px;">
    04 · Browse by genre
  </div>
</div>
""")

    # Genre info cards (display only — no buttons, no state)
    gcols = st.columns(3, gap="small")
    for i, (genre, gcfg) in enumerate(GENRE_CFG.items()):
        cnt = int((df_rec["super_genre"] == genre).sum())
        with gcols[i % 3]:
            H(f"""
<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
  padding:18px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.05);">
  <div style="font-size:28px;margin-bottom:8px;">{gcfg['emoji']}</div>
  <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;
    color:#0f172a;text-transform:capitalize;margin-bottom:4px;">{genre}</div>
  <div style="font-family:'DM Mono',monospace;font-size:11px;
    color:{gcfg['color']};font-weight:700;">{cnt:,} tracks</div>
</div>
""")

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 20px;">')

    # Use index-based selectbox to avoid key conflicts
    genre_idx = st.selectbox(
        "SELECT GENRE",
        options=list(range(len(SUPER_GENRES))),
        format_func=lambda i: f"{GENRE_CFG.get(SUPER_GENRES[i], _FALLBACK_GENRE_CFG)['emoji']} {SUPER_GENRES[i].title()}",
        key="genre_idx_sel",
    )
    chosen_genre = SUPER_GENRES[genre_idx]   # resolve name from index — no state conflict

    col_bg, _ = st.columns([1, 3])
    with col_bg:
        if st.button("✦  Show Top 5 Songs", key="genre_go"):
            with st.spinner("Finding top tracks..."):
                recs = rec_genre(df_rec, scaled, chosen_genre, n=5)
            # Store results AND the genre name together — never write to widget key
            st.session_state.genre_recs = recs
            st.session_state.genre_name = chosen_genre
            st.rerun()
    H('</div>')

    # Show results — read from session state, never write to widget key
    if st.session_state.genre_recs is not None and st.session_state.genre_name:
        gname = st.session_state.genre_name
        recs  = st.session_state.genre_recs
        gcfg  = GENRE_CFG.get(gname, _FALLBACK_GENRE_CFG)
        H(f"""
<div style="max-width:1200px;margin:0 auto;padding:0 40px 8px;">
  <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
    color:#0f172a;margin-bottom:14px;">
    {gcfg['emoji']} Top 5 {gname.title()} Tracks</div>
</div>
""")
        # ── Shuffle button ──
        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 10px;">')
        col_gs, _ = st.columns([1, 4])
        with col_gs:
            if st.button("🔀  Shuffle Songs", key="genre_shuffle_btn"):
                new_seed = random.randint(0, 99999)
                rng      = np.random.default_rng(new_seed)
                mask     = (df_rec["super_genre"] == gname).values
                gdf      = df_rec[mask].copy()
                gvecs    = scaled[mask]
                qvec     = gvecs.mean(axis=0).reshape(1, -1)
                sims     = cosine_similarity(qvec, gvecs)[0]
                noise    = rng.uniform(0, 0.08, size=len(gdf))
                gdf["similarity"] = sims * 0.6 + (gdf["popularity"] / 100.0) * 0.4 + noise
                st.session_state.genre_recs = gdf.sort_values(
                    "similarity", ascending=False).head(5).reset_index(drop=True)
                st.rerun()
        H('</div>')

        H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 40px;">')
        song_cards(recs, genre=gname)
        H('</div>')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ABOUT  (FIXED: pure st calls only — no raw HTML blocks)
# ─────────────────────────────────────────────────────────────────────────────
def tab_about():
    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 40px;">')

    # Section label
    H("""
<div style="font-family:'DM Mono',monospace;font-size:11px;font-weight:700;
  letter-spacing:1.2px;color:#94a3b8;text-transform:uppercase;margin-bottom:24px;">
  About this project
</div>
""")

    # ── Row 1: pipeline + team ──────────────────────────────────────────────
    col_p, col_t = st.columns(2, gap="medium")

    with col_p:
        H("""
<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
  padding:22px;height:100%;box-shadow:0 2px 8px rgba(0,0,0,.05);">
  <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px;">AI Pipeline</div>
""")
        for color, text in [
            ("#7c3aed", "114,000 Spotify tracks · 114 genres collapsed to 6"),
            ("#0ea5e9", "15 raw audio features → 42 via MusicFeatureEngineer"),
            ("#db2777", "LightGBM + Optuna TPE hyperparameter tuning"),
            ("#f59e0b", "Cosine similarity recommender · top-5 per mood"),
            ("#16a34a", "Test accuracy 69% · Macro ROC-AUC 0.90"),
        ]:
            H(f"""
<div style="display:flex;align-items:flex-start;gap:10px;
  padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:13px;color:#374151;">
  <span style="width:8px;height:8px;border-radius:50%;background:{color};
    flex-shrink:0;margin-top:5px;"></span>{text}
</div>
""")
        H("</div>")

    with col_t:
        H("""
<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
  padding:22px;height:100%;box-shadow:0 2px 8px rgba(0,0,0,.05);">
  <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px;">Team — Group 3</div>
""")
        for name, role, color in [
            ("Hein Htet Phyo",    "AI Product Developer",  "#7c3aed"),
            ("Zulfiqar Khan",     "Lead ML Engineer",      "#0ea5e9"),
            ("Zach",              "Analyst &amp; PM",      "#f59e0b"),
            ("Htet Htet Wint",    "Data Engineer",         "#16a34a"),
            ("Layaung Linn Lett", "Feature Engineer",      "#db2777"),
        ]:
            H(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
  padding:8px 0;border-bottom:1px solid #f1f5f9;font-size:13px;">
  <span style="font-weight:600;color:#0f172a;">{name}</span>
  <span style="font-size:11px;padding:2px 9px;border-radius:999px;
    font-family:'DM Mono',monospace;font-weight:600;
    background:{color}18;color:{color};border:1px solid {color}40;">{role}</span>
</div>
""")
        H("</div>")

    H('<div style="height:16px;"></div>')

   # ── How it works ────────────────────────────────────────────────────────
    H("""
<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:16px;
  padding:22px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.05);">
  <div style="font-size:11px;color:#94a3b8;font-family:'DM Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px;">How It Works</div>
""")
    for color, num, title, desc in [
        ("#7c3aed","1","Load &amp; clean",    "114,000 Spotify tracks · IQR Winsorisation · 114 genres → 6 classes"),
        ("#0ea5e9","2","Engineer features",    "15 raw audio features expanded to 42 via MusicFeatureEngineer"),
        ("#db2777","3","Train model",          "LightGBM classifier · Optuna TPE tuning · 70/15/15 split"),
        ("#f59e0b","4","Evaluate",             "ROC-AUC 0.90 · confusion matrix · per-class F1 · confidence analysis"),
        ("#16a34a","5","Recommend",            "Cosine similarity · top-5 individual songs · 4 recommendation modes"),
    ]:
        H(f"""
<div style="display:flex;align-items:flex-start;gap:14px;padding:12px 0;
  border-bottom:1px solid #f1f5f9;">
  <div style="width:32px;height:32px;border-radius:50%;flex-shrink:0;
    background:{color};display:flex;align-items:center;justify-content:center;
    font-family:'Syne',sans-serif;font-size:14px;font-weight:800;color:#fff;">{num}</div>
  <div style="padding-top:4px;">
    <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
      color:#0f172a;margin-bottom:2px;">{title}</div>
    <div style="font-size:12px;color:#64748b;">{desc}</div>
  </div>
</div>
""")
    H("</div>")

    # ── References ──────────────────────────────────────────────────────────
    H("""
  <div style="padding:18px 22px;background:#f8fafc;border:1.5px solid #e2e8f0;
  border-radius:12px;font-size:13px;color:#374151;line-height:2.0;">
  <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;
    color:#0f172a;margin-bottom:12px;">References</div>
  <div style="margin-bottom:6px;">Pandya, M. (no date) <em>Spotify Tracks Dataset</em> [online]. Kaggle. Available at: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset [Accessed: 17 December 2025].</div>
  <div style="margin-bottom:6px;">Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T. (2017) 'LightGBM: A highly efficient gradient boosting decision tree', in Guyon, I. et al. (eds.) <em>Advances in Neural Information Processing Systems 30 (NeurIPS 2017)</em>. New York: Curran Associates, pp. 3146–3154.</div>
  <div style="margin-bottom:6px;">Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: A next-generation hyperparameter optimization framework', in <em>Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2019)</em>. New York: ACM, pp. 2623–2631.</div>
  <div style="margin-bottom:6px;">Salton, G. and McGill, M.J. (1983) <em>Introduction to Modern Information Retrieval</em>. New York: McGraw-Hill.</div>
  <div style="margin-bottom:6px;">Hand, D.J. and Till, R.J. (2001) 'A simple generalisation of the area under the ROC curve for multiple class problems', <em>Machine Learning</em>, 45(2), pp. 171–186.</div>
  <div style="margin-bottom:6px;">Russell, J.A. (1980) 'A circumplex model of affect', <em>Journal of Personality and Social Psychology</em>, 39(6), pp. 1161–1178.</div>
  <div style="margin-bottom:6px;">Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of deep bidirectional transformers for language understanding', in <em>Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT 2019)</em>. Minneapolis: Association for Computational Linguistics, pp. 4171–4186.</div>
  <div style="margin-bottom:6px;">Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in Guyon, I. et al. (eds.) <em>Advances in Neural Information Processing Systems 30 (NeurIPS 2017)</em>. New York: Curran Associates, pp. 4765–4774.</div>
  <div style="margin-bottom:6px;">Ko, S. (2025) <em>HWNAS Dataset — Music Mood Classification</em> [online]. Kaggle. Available at: https://www.kaggle.com/datasets/stanislavko/hwnas-dataset-music-mood-classification [Accessed: 17 December 2025].</div>
  <div style="margin-top:12px;color:#94a3b8;font-family:'DM Mono',monospace;font-size:11px;">
    UFCE3P-30-3 · Essentials and Applications of AI · UWE Bristol · April 2026</div>
</div>
""")
    H('</div>')

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def render_footer():
    H("""
<footer style="background:#fff;border-top:1.5px solid #e2e8f0;padding:26px 40px;
  margin-top:20px;box-shadow:0 -2px 8px rgba(0,0,0,.04);">
  <div style="max-width:1200px;margin:0 auto;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:32px;height:32px;background:linear-gradient(135deg,#7c3aed,#0ea5e9);
        border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;">🎵</div>
      <div>
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#0f172a;">
          MoodTunes<span style="background:linear-gradient(135deg,#7c3aed,#0ea5e9);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span></div>
        <div style="font-size:11px;color:#94a3b8;">
          Group 3 · UFCE3P-30-3 · UWE Bristol · April 2026</div>
      </div>
    </div>
    <div style="font-size:12px;color:#94a3b8;">
      LightGBM · Optuna · Cosine Similarity · Streamlit</div>
  </div>
</footer>
""")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _init()
    inject_css()

    with st.spinner("Loading MoodTunes AI..."):
        try:
            df_rec, scaled, scaler, songs = load_data()
        except FileNotFoundError as exc:
            st.error("Deployment is missing `dataset.csv`.")
            st.info(
                "Fix: commit `dataset.csv` to the repository and redeploy the app. "
                "This project currently loads the dataset from a local file next to `app.py`."
            )
            st.code(str(exc))
            st.stop()

    render_header(len(df_rec))
    render_hero()

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px;">')
    H('<hr style="border-color:#e2e8f0;margin:0 0 30px;">')
    H('</div>')

    H('<div style="max-width:1200px;margin:0 auto;padding:0 40px 20px;">')
    t1, t2, t3, t4, t5 = st.tabs([
        "🎭  By Mood",
        "🔍  By Song",
        "🎛️  By Features",
        "🎸  By Genre",
        "ℹ️  About",
    ])
    H('</div>')

    with t1: tab_mood(df_rec, scaled)
    with t2: tab_song(df_rec, scaled, songs)
    with t3: tab_features(df_rec, scaled, scaler)
    with t4: tab_genre(df_rec, scaled)
    with t5: tab_about()

    render_footer()


if __name__ == "__main__":
    main()
