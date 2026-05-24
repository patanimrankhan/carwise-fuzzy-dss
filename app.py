import base64
from io import BytesIO
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="CarWise Fuzzy",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "usedcar_fuzzy_evaluation_dataset_50.csv"

# ============================================================
# Language support: compact translations for the main UI
# ============================================================
LANG = {
    "English": {
        "tagline": "Human-centred fuzzy recommendations for Swiss used-car choices.",
        "start": "Start your recommendation",
        "priority": "What should the system care about most?",
        "budget": "Budget",
        "balanced": "Balanced",
        "reliability": "Reliability",
        "purpose": "Main use",
        "context": "Driving context",
        "fuel": "Preferred fuel",
        "transmission": "Transmission",
        "budget_slider": "Maximum comfortable budget",
        "strict_budget": "Only show cars within budget",
        "run": "Find my best cars",
        "results": "Recommended shortlist",
        "explore_more": "Number of cars to show",
        "profile": "Buyer profile",
        "note": "Prototype note",
    },
    "Deutsch": {
        "tagline": "Menschzentrierte fuzzy Empfehlungen für Schweizer Gebrauchtwagen.",
        "start": "Empfehlung starten",
        "priority": "Was soll das System am stärksten beachten?",
        "budget": "Budget",
        "balanced": "Ausgewogen",
        "reliability": "Zuverlässigkeit",
        "purpose": "Hauptnutzung",
        "context": "Fahrumgebung",
        "fuel": "Bevorzugter Antrieb",
        "transmission": "Getriebe",
        "budget_slider": "Maximales komfortables Budget",
        "strict_budget": "Nur Autos im Budget anzeigen",
        "run": "Beste Autos finden",
        "results": "Empfohlene Auswahl",
        "explore_more": "Anzahl Autos anzeigen",
        "profile": "Käuferprofil",
        "note": "Prototyp-Hinweis",
    },
    "Français": {
        "tagline": "Recommandations floues centrées sur l'humain pour les voitures d'occasion en Suisse.",
        "start": "Démarrer la recommandation",
        "priority": "Que doit prioriser le système ?",
        "budget": "Budget",
        "balanced": "Équilibré",
        "reliability": "Fiabilité",
        "purpose": "Usage principal",
        "context": "Contexte de conduite",
        "fuel": "Énergie préférée",
        "transmission": "Transmission",
        "budget_slider": "Budget confortable maximum",
        "strict_budget": "Afficher seulement les voitures dans le budget",
        "run": "Trouver mes meilleures voitures",
        "results": "Sélection recommandée",
        "explore_more": "Nombre de voitures à afficher",
        "profile": "Profil acheteur",
        "note": "Note de prototype",
    },
    "Italiano": {
        "tagline": "Raccomandazioni fuzzy centrate sull'utente per auto usate in Svizzera.",
        "start": "Avvia raccomandazione",
        "priority": "Cosa deve considerare di più il sistema?",
        "budget": "Budget",
        "balanced": "Bilanciato",
        "reliability": "Affidabilità",
        "purpose": "Uso principale",
        "context": "Contesto di guida",
        "fuel": "Carburante preferito",
        "transmission": "Cambio",
        "budget_slider": "Budget massimo confortevole",
        "strict_budget": "Mostra solo auto nel budget",
        "run": "Trova le migliori auto",
        "results": "Lista raccomandata",
        "explore_more": "Numero di auto da mostrare",
        "profile": "Profilo acquirente",
        "note": "Nota prototipo",
    },
}

# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
<style>
:root {
  --ink: #182235;
  --muted: #5a6578;
  --glass: rgba(255,255,255,0.74);
  --glass-strong: rgba(255,255,255,0.90);
  --line: rgba(24,34,53,0.10);
  --blue: #3b82f6;
  --purple: #8b5cf6;
  --green: #10b981;
  --amber: #f59e0b;
  --red: #ef4444;
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 10%, rgba(59,130,246,0.22), transparent 30%),
    radial-gradient(circle at 88% 6%, rgba(139,92,246,0.20), transparent 28%),
    radial-gradient(circle at 78% 82%, rgba(16,185,129,0.13), transparent 24%),
    linear-gradient(140deg, #f8fbff 0%, #f5f0ff 48%, #f7fffb 100%);
}
[data-testid="stHeader"] { background: rgba(255,255,255,0); }
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.70);
  backdrop-filter: blur(18px);
  border-right: 1px solid var(--line);
}
.hero {
  position: relative;
  overflow: hidden;
  padding: 34px 34px 28px 34px;
  border: 1px solid rgba(255,255,255,0.75);
  border-radius: 30px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,255,255,0.54)),
    radial-gradient(circle at 80% 10%, rgba(59,130,246,0.15), transparent 25%);
  box-shadow: 0 28px 70px rgba(31,41,55,0.10);
  margin-bottom: 20px;
}
.hero:before {
  content: "";
  position: absolute;
  width: 360px;
  height: 360px;
  right: -110px;
  top: -130px;
  background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.20));
  border-radius: 999px;
  filter: blur(2px);
}
.hero-title {
  font-size: clamp(2.2rem, 5vw, 4.8rem);
  line-height: 0.95;
  letter-spacing: -0.06em;
  margin: 0 0 12px 0;
  color: var(--ink);
  font-weight: 850;
}
.hero-title span {
  background: linear-gradient(90deg, #2563eb, #7c3aed, #059669);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-sub {
  color: var(--muted);
  max-width: 760px;
  font-size: 1.05rem;
}
.chip-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
.chip {
  display:inline-flex;
  align-items:center;
  gap:7px;
  padding: 9px 13px;
  border-radius: 999px;
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(24,34,53,0.10);
  color: #2b3446;
  font-size: 0.92rem;
  box-shadow: 0 8px 18px rgba(31,41,55,0.05);
}
.glass-card {
  background: var(--glass);
  border: 1px solid rgba(255,255,255,0.72);
  border-radius: 26px;
  box-shadow: 0 20px 50px rgba(31,41,55,0.10);
  backdrop-filter: blur(18px);
  padding: 22px;
  margin-bottom: 18px;
}
.micro-title {
  font-size: .82rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: #667085;
  font-weight: 800;
  margin-bottom: 6px;
}
.option-help {
  display:grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 12px 0 4px 0;
}
.help-card {
  padding: 13px 14px;
  border-radius: 18px;
  border: 1px solid rgba(24,34,53,0.08);
  background: rgba(255,255,255,0.54);
  transition: transform .22s ease, box-shadow .22s ease, background .22s ease;
  min-height: 94px;
}
.help-card:hover {
  transform: translateY(-4px);
  background: rgba(255,255,255,0.86);
  box-shadow: 0 18px 32px rgba(31,41,55,0.10);
}
.help-card b { display:block; color:#1f2937; margin-bottom:4px; }
.help-card span { color:#667085; font-size:.90rem; }
.metric-pill {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:8px 11px;
  border-radius:999px;
  background:rgba(255,255,255,.75);
  border:1px solid rgba(24,34,53,.08);
  margin: 0 7px 7px 0;
  color:#344054;
  font-size: .90rem;
}
.car-card {
  position: relative;
  overflow: hidden;
  border-radius: 28px;
  padding: 20px 20px 18px 20px;
  margin-bottom: 14px;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.70)),
    radial-gradient(circle at 100% 0%, rgba(59,130,246,0.14), transparent 30%);
  border: 1px solid rgba(255,255,255,0.90);
  box-shadow: 0 18px 42px rgba(31,41,55,0.09);
  transition: transform .24s ease, box-shadow .24s ease;
}
.car-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 26px 70px rgba(31,41,55,0.15);
}
.rank-badge {
  position:absolute;
  top: 16px;
  right: 16px;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color:white;
  font-weight:800;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: .85rem;
}
.car-name {
  font-weight: 820;
  font-size: 1.18rem;
  color: var(--ink);
  padding-right: 92px;
}
.car-sub {
  color: #667085;
  margin-top: 4px;
  font-size: .92rem;
}
.score-ring {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  display:flex;
  align-items:center;
  justify-content:center;
  margin: 12px 0;
  background:
    radial-gradient(closest-side, white 74%, transparent 75% 100%),
    conic-gradient(#2563eb calc(var(--p) * 1%), #e5e7eb 0);
}
.score-ring span {
  font-size:1.4rem;
  font-weight: 850;
  color:#182235;
}
.label-strong { color:#047857; font-weight:800; }
.label-good { color:#2563eb; font-weight:800; }
.label-consider { color:#b45309; font-weight:800; }
.label-avoid { color:#b91c1c; font-weight:800; }
.reason {
  padding: 10px 12px;
  border-radius: 16px;
  background: rgba(248,250,252,0.86);
  border: 1px solid rgba(24,34,53,0.07);
  margin-bottom: 8px;
  color:#344054;
}
.soft-warning {
  border-left: 4px solid #f59e0b;
  background: rgba(255,251,235,0.80);
  padding: 12px 14px;
  border-radius: 16px;
  color: #78350f;
  margin: 10px 0;
}
.success-box {
  border-left: 4px solid #10b981;
  background: rgba(236,253,245,0.80);
  padding: 12px 14px;
  border-radius: 16px;
  color: #064e3b;
  margin: 10px 0;
}
.footer {
  text-align:center;
  color:#667085;
  padding: 26px;
  font-size:.92rem;
}
.stButton>button {
  border-radius: 999px !important;
  padding: 0.75rem 1.2rem !important;
  border: 0 !important;
  background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
  color: white !important;
  font-weight: 800 !important;
  box-shadow: 0 14px 28px rgba(37,99,235,0.22);
  transition: transform .2s ease, box-shadow .2s ease;
}
.stButton>button:hover {
  transform: translateY(-2px);
  box-shadow: 0 18px 36px rgba(37,99,235,0.30);
}
div[data-testid="stExpander"] {
  border-radius: 22px !important;
  border: 1px solid rgba(24,34,53,0.08) !important;
  background: rgba(255,255,255,0.72) !important;
}
@media (max-width: 900px) {
  .option-help { grid-template-columns: 1fr; }
}
</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Data functions
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if "baseline_recommendation_score" in df.columns and "recommendation_score" not in df.columns:
        df = df.rename(columns={"baseline_recommendation_score": "recommendation_score"})

    # Make labels from score so the app does not rely on a pre-given answer.
    def score_to_label(score):
        if pd.isna(score):
            return "Unknown"
        if score <= 30:
            return "Avoid"
        if score <= 55:
            return "Consider"
        if score <= 75:
            return "Recommend"
        return "Strongly Recommend"

    df["recommendation_label"] = df["recommendation_score"].apply(score_to_label)

    numeric_cols = [
        "price_chf", "mileage_km", "age_years", "price_score", "mileage_score", "age_score",
        "condition_score", "service_history_score", "accident_risk_score", "seller_trust_score",
        "value_for_money_indicator", "confidence_score", "recommendation_score",
        "budget_mode_score", "reliability_mode_score", "power_ps",
        "optional_equipment_count", "standard_equipment_count", "service_keyword_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safety component used in final profile scoring.
    df["accident_safety_score"] = 10 - df["accident_risk_score"].fillna(5)
    return df


def label_from_score(score):
    if score <= 30:
        return "Avoid"
    if score <= 55:
        return "Consider"
    if score <= 75:
        return "Recommend"
    return "Strongly Recommend"


def label_class(label):
    return {
        "Strongly Recommend": "label-strong",
        "Recommend": "label-good",
        "Consider": "label-consider",
        "Avoid": "label-avoid",
    }.get(label, "label-good")


def fmt_chf(value):
    try:
        return f"CHF {int(round(value)):,.0f}".replace(",", "'")
    except Exception:
        return "CHF n/a"


def normalize_weights(weights):
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def build_weights(priority, main_use, driving_context, reliability_need):
    if priority == "Budget":
        weights = {
            "price_score": 0.30, "mileage_score": 0.20, "age_score": 0.10,
            "condition_score": 0.10, "service_history_score": 0.10,
            "seller_trust_score": 0.08, "accident_safety_score": 0.12,
        }
    elif priority == "Reliability":
        weights = {
            "price_score": 0.08, "mileage_score": 0.12, "age_score": 0.12,
            "condition_score": 0.20, "service_history_score": 0.20,
            "seller_trust_score": 0.14, "accident_safety_score": 0.14,
        }
    else:
        weights = {
            "price_score": 0.18, "mileage_score": 0.16, "age_score": 0.13,
            "condition_score": 0.17, "service_history_score": 0.15,
            "seller_trust_score": 0.10, "accident_safety_score": 0.11,
        }

    # Human-context refinements: these change the importance of existing fuzzy inputs.
    if main_use == "City commute":
        weights["price_score"] += 0.05
        weights["mileage_score"] += 0.03
    elif main_use == "Family use":
        weights["condition_score"] += 0.04
        weights["accident_safety_score"] += 0.05
        weights["service_history_score"] += 0.03
    elif main_use == "Long-distance":
        weights["mileage_score"] += 0.04
        weights["service_history_score"] += 0.04
        weights["seller_trust_score"] += 0.02
    elif main_use == "Student budget":
        weights["price_score"] += 0.08
        weights["mileage_score"] += 0.02
    elif main_use == "Weekend trips":
        weights["condition_score"] += 0.03
        weights["age_score"] += 0.02

    if driving_context == "Hilly / snowy areas":
        weights["accident_safety_score"] += 0.04
        weights["condition_score"] += 0.03
    elif driving_context == "Highway":
        weights["service_history_score"] += 0.03
        weights["mileage_score"] += 0.02
    elif driving_context == "City":
        weights["price_score"] += 0.02

    if reliability_need == "High":
        weights["condition_score"] += 0.04
        weights["service_history_score"] += 0.04
        weights["accident_safety_score"] += 0.04
    elif reliability_need == "Low":
        weights["price_score"] += 0.04

    return normalize_weights(weights)


def option_bonus(row, fuel_pref, transmission_pref, car_type_pref, driving_context, main_use):
    bonus = 0.0
    notes = []

    fuel = str(row.get("fuel_type", "")).lower()
    trans = str(row.get("transmission", "")).lower()
    body = str(row.get("body_type", "")).lower()
    drive = str(row.get("drive_type", "")).lower()

    if fuel_pref != "No preference":
        wanted = fuel_pref.lower()
        if wanted == "hybrid":
            match = "hybrid" in fuel
        else:
            match = wanted in fuel
        if match:
            bonus += 3.0
            notes.append(f"matches preferred fuel: {fuel_pref}")
        else:
            bonus -= 1.5

    if transmission_pref != "No preference":
        if transmission_pref.lower() in trans:
            bonus += 1.5
            notes.append(f"matches preferred transmission: {transmission_pref}")
        else:
            bonus -= 1.0

    # Current dataset is sedan-focused, so body type is soft rather than hard.
    if car_type_pref != "No preference":
        if car_type_pref.lower() in body:
            bonus += 2.0
            notes.append(f"matches preferred car type: {car_type_pref}")
        elif car_type_pref == "SUV":
            bonus -= 0.5  # small penalty only; dataset is currently sedan-focused.

    if driving_context == "Hilly / snowy areas":
        if "all-wheel" in drive or "4x4" in drive:
            bonus += 4.0
            notes.append("all-wheel drive supports hilly/snowy roads")
        else:
            bonus -= 1.0

    if main_use == "City commute":
        if "electric" in fuel or "hybrid" in fuel:
            bonus += 2.0
            notes.append("electric/hybrid profile suits city commuting")
        if "automatic" in trans or "automated" in trans:
            bonus += 0.8

    if main_use == "Long-distance":
        consumption = str(row.get("consumption_raw", "")).lower()
        if "5." in consumption or "4." in consumption or "electric" in fuel or "hybrid" in fuel:
            bonus += 1.4
            notes.append("efficient profile helps long-distance use")

    return bonus, notes


def compute_profile_scores(
    df,
    priority,
    main_use,
    driving_context,
    fuel_pref,
    transmission_pref,
    car_type_pref,
    max_budget,
    strict_budget,
    reliability_need,
):
    weights = build_weights(priority, main_use, driving_context, reliability_need)
    d = df.copy()

    # Weighted fuzzy score from the seven dimensions.
    score = np.zeros(len(d))
    for col, w in weights.items():
        score += d[col].fillna(d[col].median()).clip(0, 10).to_numpy() * 10 * w

    profile_notes = []
    bonuses = []
    all_notes = []
    for _, row in d.iterrows():
        bonus, notes = option_bonus(row, fuel_pref, transmission_pref, car_type_pref, driving_context, main_use)
        bonuses.append(bonus)
        all_notes.append(notes)

    d["profile_bonus"] = bonuses
    d["profile_notes"] = all_notes
    d["profile_score"] = score + d["profile_bonus"]

    # Budget penalty or filtering.
    if strict_budget:
        d = d[d["price_chf"] <= max_budget].copy()
    else:
        over = (d["price_chf"] - max_budget).clip(lower=0)
        penalty = (over / max(max_budget, 1)) * 18
        d["profile_score"] = d["profile_score"] - penalty

    # Reliability need as minimum expectation, not a harsh binary filter.
    if reliability_need == "High":
        reliability_proxy = (
            d["condition_score"].fillna(5) +
            d["service_history_score"].fillna(5) +
            d["seller_trust_score"].fillna(5) +
            d["accident_safety_score"].fillna(5)
        ) / 4
        d["profile_score"] = d["profile_score"] + (reliability_proxy - 7.0) * 2.4

    d["profile_score"] = d["profile_score"].clip(0, 100).round(1)
    d["profile_label"] = d["profile_score"].apply(label_from_score)

    # Difference from baseline/neutral fuzzy score.
    d["fit_gain"] = (d["profile_score"] - d["recommendation_score"]).round(1)

    d = d.sort_values(["profile_score", "confidence_score", "value_for_money_indicator"], ascending=False)
    return d, weights


def make_reasons(row, priority, main_use, driving_context):
    reasons = []

    if row["condition_score"] >= 8:
        reasons.append("Strong condition evidence supports a reliable recommendation.")
    if row["service_history_score"] >= 8:
        reasons.append("Service and warranty-related evidence are strong.")
    if row["seller_trust_score"] >= 8:
        reasons.append("Seller/dealer trust evidence is high.")
    if row["accident_risk_score"] <= 2:
        reasons.append("Low accident-risk evidence improves confidence.")
    if row["mileage_score"] >= 8:
        reasons.append("Mileage is low for the evaluated market range.")
    if row["age_score"] >= 8:
        reasons.append("The car is relatively recent.")
    if row["price_score"] >= 8:
        reasons.append("Price is favourable for a budget-aware buyer.")
    if row["value_for_money_indicator"] >= 7.3:
        reasons.append("Value-for-money indicator is strong compared with similar listings.")
    if driving_context == "Hilly / snowy areas" and ("4x4" in str(row.get("drive_type", "")) or "All-wheel" in str(row.get("drive_type", ""))):
        reasons.append("All-wheel drive is useful for hilly or snowy driving contexts.")
    if main_use == "City commute" and ("Electric" in str(row.get("fuel_type", "")) or "hybrid" in str(row.get("fuel_type", "")).lower()):
        reasons.append("Electric/hybrid profile fits calm city commuting.")

    if not reasons:
        reasons.append("The recommendation is based on a balanced combination of the seven fuzzy inputs.")

    return reasons[:3]


def confidence_word(score):
    if score >= 9:
        return "High"
    if score >= 7:
        return "Medium"
    return "Limited"


def car_svg(color1="#2563eb", color2="#8b5cf6"):
    svg = f"""
    <svg viewBox="0 0 640 260" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" x2="1">
          <stop offset="0%" stop-color="{color1}"/>
          <stop offset="100%" stop-color="{color2}"/>
        </linearGradient>
      </defs>
      <rect width="640" height="260" rx="34" fill="url(#g)" opacity="0.92"/>
      <circle cx="142" cy="196" r="36" fill="#111827" opacity=".85"/>
      <circle cx="142" cy="196" r="15" fill="#f8fafc"/>
      <circle cx="500" cy="196" r="36" fill="#111827" opacity=".85"/>
      <circle cx="500" cy="196" r="15" fill="#f8fafc"/>
      <path d="M92 174 C112 120, 166 94, 236 86 L390 86 C452 88, 504 118, 545 171 L572 174 C588 176, 599 187, 596 202 L78 202 C74 187, 80 177, 92 174Z" fill="white" opacity=".92"/>
      <path d="M225 102 L300 102 L300 150 L172 150 C184 126, 200 111, 225 102Z" fill="#dbeafe"/>
      <path d="M318 102 L385 102 C423 106, 456 127, 482 150 L318 150Z" fill="#dbeafe"/>
      <path d="M114 167 L540 167" stroke="#111827" stroke-width="7" opacity=".12"/>
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def recommendation_card(row, rank):
    label = row["profile_label"]
    cls = label_class(label)
    conf = confidence_word(row.get("confidence_score", 0))
    svg_uri = car_svg()
    score = float(row["profile_score"])
    return f"""
    <div class="car-card">
      <div class="rank-badge">#{rank}</div>
      <div class="car-name">{row['car_name']}</div>
      <div class="car-sub">{row.get('fuel_type','')} • {row.get('transmission','')} • {fmt_chf(row['price_chf'])}</div>
      <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
        <div class="score-ring" style="--p:{score};"><span>{score:.0f}</span></div>
        <div>
          <div class="{cls}" style="font-size:1.05rem;">{label}</div>
          <div class="car-sub">{int(row['mileage_km']):,} km • {row['age_years']:.1f} years • confidence: {conf}</div>
          <div style="margin-top:10px;">
            <span class="metric-pill">💰 {fmt_chf(row['price_chf'])}</span>
            <span class="metric-pill">🛞 {row.get('drive_type','n/a')}</span>
            <span class="metric-pill">🧭 VFM {row.get('value_for_money_indicator',0):.1f}/10</span>
          </div>
        </div>
      </div>
    </div>
    """


def plot_mode_comparison(row):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Budget mode", "Balanced profile", "Reliability mode"],
        y=[row["budget_mode_score"], row["profile_score"], row["reliability_mode_score"]],
        text=[f"{row['budget_mode_score']:.1f}", f"{row['profile_score']:.1f}", f"{row['reliability_mode_score']:.1f}"],
        textposition="outside",
        marker=dict(color=["#3b82f6", "#8b5cf6", "#10b981"]),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=24, b=20),
        yaxis=dict(range=[0, 105], title="Score"),
        showlegend=False,
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def render_option_help(title, cards):
    card_html = "".join(
        f"""<div class="help-card"><b>{emoji} {name}</b><span>{desc}</span></div>"""
        for emoji, name, desc in cards
    )
    st.markdown(f"""<div class="micro-title">{title}</div><div class="option-help">{card_html}</div>""", unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================
df = load_data()

with st.sidebar:
    language = st.selectbox("Language / Sprache / Langue / Lingua", list(LANG.keys()), index=0)
    T = LANG[language]
    st.markdown("### 🚗 CarWise Fuzzy")
    st.caption("A fuzzy DSS prototype for used-car purchase support.")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"{len(df)} cars • 30 real AutoScout24 samples • 20 synthetic balancing cases")
    st.markdown("**Model**")
    st.caption("7 fuzzy inputs + buyer profile weighting + explanation reasons")
    st.markdown("---")
    st.caption("Tip: use the left settings first, then scroll to the shortlist.")

T = LANG[language]

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">CarWise <span>Fuzzy</span></div>
  <div class="hero-sub">{T['tagline']} Instead of forcing hard filters, this prototype converts vague human preferences into explainable fuzzy recommendations.</div>
  <div class="chip-row">
    <div class="chip">🧠 Fuzzy logic</div>
    <div class="chip">🚦 Ranked shortlist</div>
    <div class="chip">🔍 Explanation reasons</div>
    <div class="chip">🛡 Confidence indicator</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([0.38, 0.62], gap="large")

with left:
    st.markdown(f"""<div class="glass-card"><div class="micro-title">{T['start']}</div>""", unsafe_allow_html=True)

    render_option_help(
        "Priority guide",
        [
            ("💸", "Budget", "Lower price and value-for-money matter most."),
            ("⚖️", "Balanced", "A calm compromise between cost and reliability."),
            ("🛡️", "Reliability", "Safer, newer, lower-risk cars are preferred."),
        ],
    )
    priority = st.radio(
        T["priority"],
        ["Budget", "Balanced", "Reliability"],
        horizontal=True,
        index=1,
        help="This changes the weight of the seven fuzzy inputs. It does not replace the fuzzy model.",
    )

    render_option_help(
        "Use-case guide",
        [
            ("🏙️", "City commute", "Short daily routes, parking, traffic, and easy driving."),
            ("👨‍👩‍👧", "Family use", "Safety, comfort, and confidence matter more."),
            ("🛣️", "Long-distance", "Mileage, service evidence, and efficiency matter more."),
        ],
    )
    main_use = st.selectbox(
        T["purpose"],
        ["City commute", "Family use", "Long-distance", "Student budget", "Weekend trips"],
        index=0,
        help="This helps the model adjust the importance of price, mileage, condition, and reliability evidence.",
    )

    render_option_help(
        "Driving context",
        [
            ("🚦", "City", "More stop-and-go driving and short trips."),
            ("🛣️", "Highway", "Longer stable driving where service and comfort matter."),
            ("🏔️", "Hilly / snowy", "More value for 4x4, condition, and safety evidence."),
        ],
    )
    driving_context = st.selectbox(
        T["context"],
        ["City", "Highway", "Mixed", "Hilly / snowy areas"],
        index=2,
        help="This is used as a soft preference, not as a hard filter.",
    )

    fuel_pref = st.selectbox(
        T["fuel"],
        ["No preference", "Petrol", "Diesel", "Electric", "Hybrid"],
        index=0,
        help="The app gives a small bonus to matching fuel types. If no match exists, it still shows the best alternatives.",
    )
    transmission_pref = st.selectbox(
        T["transmission"],
        ["No preference", "Automatic", "Manual"],
        index=0,
        help="Automatic can be useful in city traffic; manual may fit budget-focused users.",
    )
    car_type_pref = st.selectbox(
        "Preferred car type",
        ["No preference", "Sedan", "SUV", "Hatchback"],
        index=0,
        help="Current prototype data is sedan-focused; this preference is treated softly unless matching data exists.",
    )
    max_budget = st.slider(
        T["budget_slider"],
        min_value=4000,
        max_value=90000,
        value=35000,
        step=1000,
        help="Budget can be strict or soft depending on the checkbox below.",
    )
    strict_budget = st.checkbox(T["strict_budget"], value=False)
    reliability_need = st.select_slider(
        "Minimum reliability expectation",
        options=["Low", "Medium", "High"],
        value="Medium",
        help="High reliability increases the importance of condition, service history, accident safety, and seller trust.",
    )

    show_n = st.slider(T["explore_more"], min_value=3, max_value=20, value=10, step=1)

    run = st.button(T["run"], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### {T['profile']}")
    st.markdown(
        f"""
<span class="metric-pill">🎯 {priority}</span>
<span class="metric-pill">🧭 {main_use}</span>
<span class="metric-pill">🌍 {driving_context}</span>
<span class="metric-pill">⛽ {fuel_pref}</span>
<span class="metric-pill">💰 {fmt_chf(max_budget)}</span>
""",
        unsafe_allow_html=True,
    )

    if car_type_pref in ["SUV", "Hatchback"]:
        st.markdown(
            """
<div class="soft-warning">
Current prototype data is mainly based on sedan listings. The selected body type is therefore treated as a soft preference. 
The model still ranks the available cars honestly and avoids pretending to have SUV/hatchback coverage that is not in the dataset.
</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="success-box">
The app keeps the original seven fuzzy dimensions. Extra buyer questions only adjust the weighting or filtering of those dimensions, so the final implementation remains consistent with the evaluation deliverable.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    ranked, weights = compute_profile_scores(
        df,
        priority,
        main_use,
        driving_context,
        fuel_pref,
        transmission_pref,
        car_type_pref,
        max_budget,
        strict_budget,
        reliability_need,
    )

    if strict_budget and ranked.empty:
        st.warning("No cars are available within the strict budget. Disable strict budget or increase the budget slider.")
    else:
        st.markdown(f"## {T['results']}")

        st.markdown(
            f"""
<div class="glass-card">
  <div class="micro-title">How the profile was translated</div>
  <span class="metric-pill">Price weight: {weights['price_score']:.0%}</span>
  <span class="metric-pill">Mileage weight: {weights['mileage_score']:.0%}</span>
  <span class="metric-pill">Condition weight: {weights['condition_score']:.0%}</span>
  <span class="metric-pill">Service weight: {weights['service_history_score']:.0%}</span>
  <span class="metric-pill">Accident safety weight: {weights['accident_safety_score']:.0%}</span>
</div>
            """,
            unsafe_allow_html=True,
        )

        top = ranked.head(show_n).copy()

        for idx, (_, row) in enumerate(top.iterrows(), start=1):
            st.markdown(recommendation_card(row, idx), unsafe_allow_html=True)

            with st.expander(f"See why #{idx}: {row['car_name']}"):
                c1, c2 = st.columns([0.50, 0.50], gap="large")
                with c1:
                    st.markdown("#### Top explanation reasons")
                    for reason in make_reasons(row, priority, main_use, driving_context):
                        st.markdown(f'<div class="reason">✅ {reason}</div>', unsafe_allow_html=True)

                    st.markdown("#### Fuzzy input snapshot")
                    snapshot = pd.DataFrame(
                        {
                            "Fuzzy dimension": [
                                "Price", "Mileage", "Age", "Condition",
                                "Service history", "Accident safety", "Seller trust",
                            ],
                            "Score": [
                                row["price_score"], row["mileage_score"], row["age_score"],
                                row["condition_score"], row["service_history_score"],
                                row["accident_safety_score"], row["seller_trust_score"],
                            ],
                        }
                    )
                    st.dataframe(snapshot, hide_index=True, use_container_width=True)

                with c2:
                    st.markdown("#### Buyer-mode comparison")
                    st.plotly_chart(plot_mode_comparison(row), use_container_width=True)

                    st.markdown("#### Listing evidence")
                    st.markdown(
                        f"""
- **Fuel:** {row.get('fuel_type','n/a')}
- **Transmission:** {row.get('transmission','n/a')}
- **Drive:** {row.get('drive_type','n/a')}
- **MFK included:** {row.get('mfk_included','n/a')}
- **Warranty:** {row.get('warranty','n/a')}
- **Accident damage:** {row.get('accident_damage','n/a')}
- **Confidence/data completeness:** {row.get('confidence_score',0):.1f}/10
                        """
                    )

        st.markdown("## Explore the ranked data")
        visible_cols = [
            "car_id", "car_name", "price_chf", "mileage_km", "age_years",
            "fuel_type", "transmission", "profile_score", "profile_label",
            "confidence_score", "value_for_money_indicator",
        ]
        st.dataframe(
            ranked[visible_cols].head(show_n).rename(columns={
                "profile_score": "final_score",
                "profile_label": "final_label",
            }),
            hide_index=True,
            use_container_width=True,
        )

        csv = ranked[visible_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranked recommendations as CSV",
            data=csv,
            file_name="carwise_fuzzy_ranked_recommendations.csv",
            mime="text/csv",
        )

st.markdown(
    """
<div class="footer">
CarWise Fuzzy • Final implementation prototype • Fuzzy DSS for used-car purchase evaluation
</div>
    """,
    unsafe_allow_html=True,
)
