import streamlit as st
import numpy as np
from PIL import Image
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSmart — Diagnostic IA",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES (mis en cache — chargé une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Charge les deux modèles Keras au démarrage.
    Les fichiers doivent être dans le même dossier que app.py :
        cnn_best.keras        → Manioc
        cnn_simple_best.keras → Maïs
    """
    import tensorflow as tf

    base = os.path.dirname(os.path.abspath(__file__))
    path_manioc = os.path.join(base, "cnn_best.keras")
    path_mais   = os.path.join(base, "cnn_simple_best.keras")

    errors = []
    if not os.path.exists(path_manioc):
        errors.append(f"❌ Modèle Manioc introuvable : {path_manioc}")
    if not os.path.exists(path_mais):
        errors.append(f"❌ Modèle Maïs introuvable : {path_mais}")
    if errors:
        raise FileNotFoundError("\n".join(errors))

    model_manioc = tf.keras.models.load_model(path_manioc, compile=False)
    model_mais   = tf.keras.models.load_model(path_mais, compile=False)
    return {"manioc": model_manioc, "mais": model_mais}

# ─────────────────────────────────────────────────────────────────────────────
# LANGUE
# ─────────────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

TEXTS = {
    "FR": {
        "loading":      "Chargement des modèles IA...",
        "model_ready":  "Modèles chargés",
        "tagline":      "Diagnostic agricole par Intelligence Artificielle",
        "subtitle":     "Photographiez une feuille, obtenez un diagnostic en 2 secondes",
        "choose_crop":  "Choisissez votre culture",
        "maize":        "Maïs",
        "cassava":      "Manioc",
        "maize_desc":   "4 maladies détectées · 93.2% de précision",
        "cassava_desc": "5 maladies détectées · ~72% de précision",
        "upload_title": "Uploadez la photo de la feuille",
        "upload_hint":  "Formats acceptés : JPG, JPEG, PNG",
        "analyze_btn":  "🔬 Analyser",
        "analyzing":    "Analyse en cours...",
        "result_title": "Résultat du diagnostic",
        "confidence":   "Confiance",
        "treatment":    "Traitement recommandé",
        "prevention":   "Prévention",
        "severity":     "Sévérité",
        "reset":        "Nouveau diagnostic",
        "how_title":    "Comment ça marche ?",
        "step1":        "Choisissez la culture",
        "step2":        "Uploadez la photo",
        "step3":        "Obtenez le diagnostic",
        "step1_desc":   "Sélectionnez maïs ou manioc",
        "step2_desc":   "Photo de la feuille affectée",
        "step3_desc":   "Résultat instantané + traitement",
        "model_info":   "Informations du modèle",
        "accuracy":     "Précision",
        "classes_lbl":  "Classes",
        "size_lbl":     "Architecture",
        "tip_text":     "Photo sur fond neutre, lumière naturelle, à 20-30 cm. Centrez bien la feuille.",
        "note":         "⚡ Modèle CNN Keras · Inférence GPU/CPU · Temps réel",
        "prob_title":   "Distribution des probabilités",
        "lang_btn":     "🇬🇧 English",
        "error_model":  "Erreur de chargement des modèles",
    },
    "EN": {
        "loading":      "Loading AI models...",
        "model_ready":  "Models loaded",
        "tagline":      "Agricultural Diagnosis by Artificial Intelligence",
        "subtitle":     "Photograph a leaf, get a diagnosis in 2 seconds",
        "choose_crop":  "Choose your crop",
        "maize":        "Maize",
        "cassava":      "Cassava",
        "maize_desc":   "4 diseases detected · 93.2% accuracy",
        "cassava_desc": "5 diseases detected · ~72% accuracy",
        "upload_title": "Upload the leaf photo",
        "upload_hint":  "Accepted formats: JPG, JPEG, PNG",
        "analyze_btn":  "🔬 Analyze",
        "analyzing":    "Analyzing...",
        "result_title": "Diagnosis Result",
        "confidence":   "Confidence",
        "treatment":    "Recommended Treatment",
        "prevention":   "Prevention",
        "severity":     "Severity",
        "reset":        "New Diagnosis",
        "how_title":    "How does it work?",
        "step1":        "Choose the crop",
        "step2":        "Upload the photo",
        "step3":        "Get the diagnosis",
        "step1_desc":   "Select maize or cassava",
        "step2_desc":   "Photo of the affected leaf",
        "step3_desc":   "Instant result + treatment advice",
        "model_info":   "Model Information",
        "accuracy":     "Accuracy",
        "classes_lbl":  "Classes",
        "size_lbl":     "Architecture",
        "tip_text":     "Photo on neutral background, natural light, 20-30 cm away. Center the leaf.",
        "note":         "⚡ CNN Keras model · GPU/CPU inference · Real-time",
        "prob_title":   "Probability Distribution",
        "lang_btn":     "🇫🇷 Français",
        "error_model":  "Model loading error",
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES MALADIES
# ─────────────────────────────────────────────────────────────────────────────
DISEASES = {
    "mais": {
        "model_file": "cnn_simple_best.keras",
        "classes":    ["sain", "blight", "rouille", "cercosporiose"],
        "accuracy":   "93.2%",
        "arch":       "CNN Scratch · 3 blocs Conv",
        "icon":       "🌽",
        "color":      "#F39C12",
        "FR": {
            "sain":          {"name": "Feuille Saine",           "emoji": "✅", "color": "#27AE60", "severity": "Aucune",     "bg": "#0A1A0A", "treatment": "Aucun traitement nécessaire. Continuez les bonnes pratiques agricoles.", "prevention": "Rotation des cultures, semences certifiées, espacement adéquat."},
            "blight":        {"name": "Brûlure Foliaire (Blight)","emoji": "🟤", "color": "#A0522D", "severity": "Élevée",     "bg": "#1A1000", "treatment": "Fongicide mancozèbe ou chlorothalonil. Éliminez les résidus infectés.", "prevention": "Variétés résistantes, évitez l'excès d'humidité."},
            "rouille":       {"name": "Rouille Commune",          "emoji": "🟠", "color": "#E67E22", "severity": "Moyenne",    "bg": "#1A0F00", "treatment": "Fongicides triazoles (propiconazole). Traitement précoce recommandé.", "prevention": "Semis précoce, variétés hybrides résistantes."},
            "cercosporiose": {"name": "Cercosporiose",            "emoji": "🔴", "color": "#E74C3C", "severity": "Élevée",     "bg": "#1A0000", "treatment": "Fongicides strobilurine. Éliminez les feuilles infectées.", "prevention": "Rotation biennale, labour des résidus de culture."},
        },
        "EN": {
            "sain":          {"name": "Healthy Leaf",        "emoji": "✅", "color": "#27AE60", "severity": "None",    "bg": "#0A1A0A", "treatment": "No treatment needed. Continue good agricultural practices.", "prevention": "Crop rotation, certified seeds, adequate spacing."},
            "blight":        {"name": "Northern Leaf Blight","emoji": "🟤", "color": "#A0522D", "severity": "High",    "bg": "#1A1000", "treatment": "Apply mancozeb or chlorothalonil fungicide. Remove infected residues.", "prevention": "Use resistant varieties, avoid excess humidity."},
            "rouille":       {"name": "Common Rust",         "emoji": "🟠", "color": "#E67E22", "severity": "Medium",  "bg": "#1A0F00", "treatment": "Triazole fungicides (propiconazole). Early treatment recommended.", "prevention": "Early sowing, resistant hybrid varieties."},
            "cercosporiose": {"name": "Gray Leaf Spot",      "emoji": "🔴", "color": "#E74C3C", "severity": "High",    "bg": "#1A0000", "treatment": "Strobilurin-based fungicides. Remove infected leaves.", "prevention": "Biennial rotation, tillage of crop residues."},
        },
    },
    "manioc": {
        "model_file": "cnn_best.keras",
        "classes":    ["sain", "cmd", "cbb", "cgm", "cbsd"],
        "accuracy":   "~72%",
        "arch":       "CNN Scratch · 3 blocs Conv",
        "icon":       "🌿",
        "color":      "#27AE60",
        "FR": {
            "sain":  {"name": "Feuille Saine",             "emoji": "✅", "color": "#27AE60", "severity": "Aucune",     "bg": "#0A1A0A", "treatment": "Aucun traitement nécessaire.", "prevention": "Boutures saines, espacement correct, désherbage régulier."},
            "cmd":   {"name": "Mosaïque du Manioc (CMD)",  "emoji": "🟡", "color": "#D4AC0D", "severity": "Très élevée","bg": "#1A1A00", "treatment": "Arrachez et brûlez les plants atteints immédiatement. Pas de traitement chimique efficace.", "prevention": "Boutures certifiées CMD-résistantes. Contrôlez les aleurodes vecteurs."},
            "cbb":   {"name": "Bactériose (CBB)",          "emoji": "🔵", "color": "#2980B9", "severity": "Élevée",     "bg": "#00101A", "treatment": "Éliminez les parties infectées. Traitez avec du cuivre. Désinfectez les outils.", "prevention": "Évitez les blessures mécaniques, utilisez des boutures saines."},
            "cgm":   {"name": "Acaroïdes Verts (CGM)",    "emoji": "🟣", "color": "#8E44AD", "severity": "Moyenne",    "bg": "#0F001A", "treatment": "Acaricides biologiques (huile de neem). Introduisez des prédateurs naturels.", "prevention": "Semis en début de saison, variétés tolérantes."},
            "cbsd":  {"name": "Blight (CBSD)",             "emoji": "🔴", "color": "#E74C3C", "severity": "Très élevée","bg": "#1A0000", "treatment": "Aucun traitement. Arrachez immédiatement. Ne replantez pas sur la même parcelle.", "prevention": "Boutures certifiées CBSD-saines, contrôle des mouches blanches."},
        },
        "EN": {
            "sain":  {"name": "Healthy Leaf",                    "emoji": "✅", "color": "#27AE60", "severity": "None",       "bg": "#0A1A0A", "treatment": "No treatment needed.", "prevention": "Healthy cuttings, proper spacing, regular weeding."},
            "cmd":   {"name": "Cassava Mosaic Disease (CMD)",     "emoji": "🟡", "color": "#D4AC0D", "severity": "Very High",  "bg": "#1A1A00", "treatment": "Uproot and burn affected plants immediately. No effective chemical treatment.", "prevention": "CMD-resistant certified cuttings. Control whitefly vectors."},
            "cbb":   {"name": "Cassava Bacterial Blight (CBB)",   "emoji": "🔵", "color": "#2980B9", "severity": "High",       "bg": "#00101A", "treatment": "Remove infected parts. Treat with copper. Disinfect tools.", "prevention": "Avoid mechanical injuries, use healthy cuttings."},
            "cgm":   {"name": "Cassava Green Mite (CGM)",         "emoji": "🟣", "color": "#8E44AD", "severity": "Medium",     "bg": "#0F001A", "treatment": "Biological acaricides (neem oil). Introduce natural predators.", "prevention": "Early season planting, tolerant varieties."},
            "cbsd":  {"name": "Cassava Brown Streak (CBSD)",      "emoji": "🔴", "color": "#E74C3C", "severity": "Very High",  "bg": "#1A0000", "treatment": "No treatment. Uproot immediately. Do not replant on the same plot.", "prevention": "CBSD-certified cuttings, whitefly control."},
        },
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE RÉELLE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(image: Image.Image, crop: str, models: dict) -> dict:
    """Prétraite l'image et lance l'inférence sur le vrai modèle Keras."""
    img = image.convert("RGB").resize((160, 160))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    model  = models[crop]
    preds  = np.array(model(arr, training=False))[0]
    preds  = np.clip(preds, 0, 1)
    total  = preds.sum()
    if total > 0:
        preds = preds / total        # normaliser au cas où

    classes   = DISEASES[crop]["classes"]
    pred_idx  = int(np.argmax(preds))

    return {
        "class":      classes[pred_idx],
        "confidence": float(preds[pred_idx]),
        "probs":      dict(zip(classes, [float(p) for p in preds]))
    }

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: #080D08 !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton { display: none !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HERO ── */
.hero {
    background: #0A0F0A;
    border-bottom: 1px solid rgba(39,174,96,0.15);
    padding: 44px 64px 36px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-40%; left:-10%;
    width:700px; height:700px;
    background: radial-gradient(circle, rgba(39,174,96,0.07) 0%, transparent 65%);
    pointer-events:none;
}
.hero-row { display:flex; justify-content:space-between; align-items:flex-start; }
.hero-left { flex:1; }
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(39,174,96,0.08); border:1px solid rgba(39,174,96,0.25);
    border-radius:100px; padding:5px 16px; margin-bottom:20px;
    font-size:11px; color:#5DADE2; letter-spacing:2px; text-transform:uppercase;
}
.hero-badge .dot {
    width:6px; height:6px; border-radius:50%; background:#27AE60;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

.hero-title {
    font-family:'Syne',sans-serif; font-size:clamp(40px,6vw,72px);
    font-weight:800; color:#fff; letter-spacing:-3px; line-height:0.95;
    margin-bottom:16px;
}
.hero-title em {
    font-style:normal;
    background:linear-gradient(135deg,#27AE60 0%,#82E0AA 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub {
    font-size:16px; color:rgba(255,255,255,0.45);
    max-width:480px; line-height:1.65; font-weight:300; margin-bottom:32px;
}
.hero-metrics { display:flex; gap:36px; }
.metric { display:flex; flex-direction:column; gap:3px; }
.metric-val {
    font-family:'Syne',sans-serif; font-size:30px;
    font-weight:800; color:#27AE60; line-height:1;
}
.metric-lbl { font-size:10px; color:rgba(255,255,255,0.3); text-transform:uppercase; letter-spacing:1.5px; }
.hero-right { display:flex; flex-direction:column; align-items:flex-end; gap:12px; padding-top:4px; }
.status-chip {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(39,174,96,0.1); border:1px solid rgba(39,174,96,0.2);
    border-radius:10px; padding:8px 16px;
    font-size:12px; color:rgba(255,255,255,0.55);
}
.status-chip .ok { color:#27AE60; font-weight:600; }

/* ── SECTION WRAPPER ── */
.section { padding: 44px 64px; background:#080D08; }
.section-label { font-size:10px; color:#27AE60; text-transform:uppercase; letter-spacing:3px; margin-bottom:6px; }
.section-title { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fff; letter-spacing:-1px; margin-bottom:28px; }

/* ── CROP CARDS ── */
.crop-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:0; }
.crop-card {
    border-radius:18px; padding:30px 28px; cursor:pointer;
    border:2px solid transparent; transition:all 0.25s ease;
    position:relative; overflow:hidden;
}
.crop-card:hover { transform:translateY(-3px); }
.crop-mais   { background:linear-gradient(145deg,#120E00,#1A1400); border-color:rgba(243,156,18,0.15); }
.crop-manioc { background:linear-gradient(145deg,#001208,#001A0A); border-color:rgba(39,174,96,0.15); }
.crop-mais:hover   { border-color:rgba(243,156,18,0.4); box-shadow:0 16px 50px rgba(243,156,18,0.12); }
.crop-manioc:hover { border-color:rgba(39,174,96,0.4);  box-shadow:0 16px 50px rgba(39,174,96,0.12); }
.crop-sel-mais   { border-color:#F39C12 !important; box-shadow:0 0 0 1px #F39C12, 0 16px 50px rgba(243,156,18,0.2) !important; }
.crop-sel-manioc { border-color:#27AE60 !important; box-shadow:0 0 0 1px #27AE60, 0 16px 50px rgba(39,174,96,0.2) !important; }
.crop-icon  { font-size:44px; display:block; margin-bottom:14px; line-height:1; }
.crop-name  { font-family:'Syne',sans-serif; font-size:24px; font-weight:700; color:#fff; margin-bottom:5px; }
.crop-desc  { font-size:12px; color:rgba(255,255,255,0.38); margin-bottom:18px; line-height:1.5; }
.pill {
    display:inline-block; padding:4px 12px; border-radius:100px;
    font-size:11px; font-weight:500; letter-spacing:0.3px;
}

/* ── UPLOAD ── */
.upload-label { font-family:'Syne',sans-serif; font-size:20px; font-weight:700; color:#fff; margin-bottom:6px; }
.upload-sub   { font-size:12px; color:rgba(255,255,255,0.35); margin-bottom:20px; }
.tip {
    display:flex; gap:10px; align-items:flex-start;
    background:rgba(243,156,18,0.05); border:1px solid rgba(243,156,18,0.15);
    border-radius:10px; padding:12px 16px; margin-bottom:20px;
}
.tip-icon { font-size:14px; flex-shrink:0; margin-top:1px; }
.tip-txt  { font-size:12px; color:rgba(255,255,255,0.45); line-height:1.55; }

/* ── RESULT ── */
.res-card { border-radius:20px; overflow:hidden; border:1px solid rgba(255,255,255,0.05); }
.res-head {
    padding:24px 28px; display:flex; align-items:center; gap:18px;
    border-bottom:1px solid rgba(255,255,255,0.05);
}
.res-emoji { font-size:44px; line-height:1; }
.res-name  { font-family:'Syne',sans-serif; font-size:24px; font-weight:800; color:#fff; margin-bottom:3px; }
.res-conf  { font-size:13px; color:rgba(255,255,255,0.4); }
.res-conf strong { font-size:15px; font-weight:600; }
.res-body  { background:#0D120D; padding:24px 28px; }
.res-lbl   { font-size:9px; text-transform:uppercase; letter-spacing:2px; color:rgba(255,255,255,0.25); margin-bottom:5px; margin-top:18px; }
.res-lbl:first-child { margin-top:0; }
.res-val   { font-size:13px; color:rgba(255,255,255,0.65); line-height:1.6; }
.sev-badge { display:inline-block; padding:3px 11px; border-radius:100px; font-size:11px; font-weight:600; }

/* ── PROB BARS ── */
.probs-wrap { margin-top:20px; }
.probs-title { font-size:9px; text-transform:uppercase; letter-spacing:2px; color:rgba(255,255,255,0.22); margin-bottom:12px; }
.pb-row  { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.pb-lbl  { font-size:11px; color:rgba(255,255,255,0.38); min-width:72px; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pb-bg   { flex:1; height:5px; background:rgba(255,255,255,0.05); border-radius:100px; overflow:hidden; }
.pb-fill { height:100%; border-radius:100px; }
.pb-pct  { font-size:11px; color:rgba(255,255,255,0.3); min-width:32px; text-align:right; }

/* ── INFO CARD ── */
.info-card {
    background:rgba(255,255,255,0.015); border:1px solid rgba(255,255,255,0.05);
    border-radius:18px; padding:26px; margin-top:4px;
}
.info-row { margin-bottom:20px; }
.info-row:last-child { margin-bottom:0; }
.info-lbl { font-size:9px; text-transform:uppercase; letter-spacing:2px; color:rgba(255,255,255,0.22); margin-bottom:4px; }
.info-val { font-family:'Syne',sans-serif; font-size:32px; font-weight:800; line-height:1; }
.info-val-sm { font-size:13px; color:rgba(255,255,255,0.5); line-height:1.6; }

/* ── HOW IT WORKS ── */
.how-section {
    padding:60px 64px;
    background:linear-gradient(180deg,#080D08,#0D1A0D);
    border-top:1px solid rgba(39,174,96,0.08);
}
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-top:32px; }
.step {
    background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05);
    border-radius:18px; padding:26px; position:relative; overflow:hidden;
}
.step::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#27AE60,transparent);
}
.step-n   { font-family:'Syne',sans-serif; font-size:44px; font-weight:800; color:rgba(39,174,96,0.1); line-height:1; margin-bottom:10px; }
.step-t   { font-family:'Syne',sans-serif; font-size:17px; font-weight:700; color:#fff; margin-bottom:6px; }
.step-d   { font-size:12px; color:rgba(255,255,255,0.35); line-height:1.6; }

/* ── FOOTER ── */
.footer {
    padding:28px 64px; border-top:1px solid rgba(255,255,255,0.04);
    display:flex; justify-content:space-between; align-items:center;
    background:#060B06;
}
.footer-brand { font-family:'Syne',sans-serif; font-size:17px; font-weight:800; color:#27AE60; }
.footer-copy  { font-size:11px; color:rgba(255,255,255,0.18); }

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background:linear-gradient(135deg,#27AE60,#1E8449) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; padding:11px 24px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:14px !important; width:100% !important; letter-spacing:0.3px !important;
    transition:all 0.2s !important;
}
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 24px rgba(39,174,96,0.3) !important; }
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,0.015) !important;
    border:2px dashed rgba(39,174,96,0.2) !important;
    border-radius:14px !important; padding:4px !important;
}
[data-testid="stFileUploader"]:hover { border-color:rgba(39,174,96,0.45) !important; }
.stSpinner > div { border-top-color:#27AE60 !important; }
.stProgress > div > div > div { background:linear-gradient(90deg,#27AE60,#82E0AA) !important; }
[data-testid="stImage"] img { border-radius:14px !important; border:1px solid rgba(255,255,255,0.07) !important; }
p, li, label, div { color:rgba(255,255,255,0.65) !important; }
h1,h2,h3 { color:#fff !important; }
hr { border-color:rgba(255,255,255,0.04) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLES AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────
T = TEXTS[st.session_state.lang]

try:
    with st.spinner(T["loading"]):
        models = load_models()
    model_status_html = f'<span class="ok">✓ {T["model_ready"]}</span> · Maïs + Manioc'
except FileNotFoundError as e:
    st.markdown(f"""
    <div style="background:#1A0000;border:1px solid #E74C3C;border-radius:14px;padding:24px;margin:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#E74C3C;margin-bottom:12px;">
            ❌ {T['error_model']}
        </div>
        <pre style="font-size:12px;color:rgba(255,255,255,0.6);white-space:pre-wrap;">{str(e)}</pre>
        <div style="font-size:12px;color:rgba(255,255,255,0.4);margin-top:12px;">
            Placez <code>cnn_best.keras</code> et <code>cnn_simple_best.keras</code> dans le même dossier que <code>app.py</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
lang_col, _ = st.columns([1, 12])
with lang_col:
    if st.button(T["lang_btn"], key="lang_btn"):
        st.session_state.lang = "EN" if st.session_state.lang == "FR" else "FR"
        st.rerun()

st.markdown(f"""
<div class="hero">
    <div class="hero-row">
        <div class="hero-left">
            <div class="hero-badge"><span class="dot"></span>D-CLIC · OIF · 2025</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">{T['tagline']}</div>
            <div class="hero-title">Agri<em>Smart</em></div>
            <div class="hero-sub">{T['subtitle']}</div>
            <div class="hero-metrics">
                <div class="metric"><div class="metric-val">93.2%</div><div class="metric-lbl">{T['maize']}</div></div>
                <div class="metric"><div class="metric-val">~72%</div><div class="metric-lbl">{T['cassava']}</div></div>
                <div class="metric"><div class="metric-val">2s</div><div class="metric-lbl">Inference</div></div>
                <div class="metric"><div class="metric-val">9</div><div class="metric-lbl">Diseases</div></div>
            </div>
        </div>
        <div class="hero-right">
            <div class="status-chip">{model_status_html}</div>
            <div class="status-chip">⚡ Keras · TensorFlow · Real-time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "crop"   not in st.session_state: st.session_state.crop   = None
if "result" not in st.session_state: st.session_state.result = None

# ─────────────────────────────────────────────────────────────────────────────
# SÉLECTION CULTURE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="section" style="padding-bottom:0;">
    <div class="section-label">{T['choose_crop']}</div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

for col, crop_key, card_cls, sel_cls, pill_color in [
    (c1, "mais",   "crop-mais",   "crop-sel-mais",   "rgba(243,156,18,0.12);color:#F39C12;border:1px solid rgba(243,156,18,0.3)"),
    (c2, "manioc", "crop-manioc", "crop-sel-manioc", "rgba(39,174,96,0.12);color:#27AE60;border:1px solid rgba(39,174,96,0.3)"),
]:
    info = DISEASES[crop_key]
    lang = st.session_state.lang
    name = T["maize"] if crop_key == "mais" else T["cassava"]
    desc = T["maize_desc"] if crop_key == "mais" else T["cassava_desc"]
    sel  = sel_cls if st.session_state.crop == crop_key else ""
    chk  = "✓ " if st.session_state.crop == crop_key else ""

    with col:
        st.markdown(f"""
        <div class="crop-card {card_cls} {sel}">
            <span class="crop-icon">{info['icon']}</span>
            <div class="crop-name">{name}</div>
            <div class="crop-desc">{desc}</div>
            <span class="pill" style="background:{pill_color};">{info['accuracy']} · {len(info['classes'])} classes</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"{chk}{name}", key=f"btn_{crop_key}"):
            st.session_state.crop   = crop_key
            st.session_state.result = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD + INFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.crop:
    crop = st.session_state.crop
    lang = st.session_state.lang
    info = DISEASES[crop]

    st.markdown(f"""
    <div class="section" style="padding-bottom:0;padding-top:36px;">
        <div class="upload-label">{info['icon']} {T['upload_title']}</div>
        <div class="upload-sub">{T['upload_hint']} · {T['note']}</div>
        <div class="tip">
            <span class="tip-icon">💡</span>
            <span class="tip-txt">{T['tip_text']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    # ── COLONNE GAUCHE : Upload ──────────────────────────────────────────────
    with left:
        st.markdown('<div style="padding:0 0 0 64px;">', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"],
                                    key=f"up_{crop}", label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if st.button(T["analyze_btn"], key="analyze"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.008)
                    progress.progress(i + 1)
                with st.spinner(T["analyzing"]):
                    result = run_inference(image, crop, models)
                st.session_state.result = result
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── COLONNE DROITE : Résultat ou Info modèle ─────────────────────────────
    with right:
        st.markdown('<div style="padding:0 64px 0 0;">', unsafe_allow_html=True)

        if st.session_state.result:
            r    = st.session_state.result
            d    = info[lang][r["class"]]
            conf = int(r["confidence"] * 100)

            st.markdown(f"""
            <div class="res-card">
                <div class="res-head" style="background:{d['bg']};">
                    <span class="res-emoji">{d['emoji']}</span>
                    <div>
                        <div class="res-name" style="color:{d['color']};">{d['name']}</div>
                        <div class="res-conf">{T['confidence']} : <strong style="color:{d['color']};">{conf}%</strong></div>
                    </div>
                </div>
                <div class="res-body">
                    <div class="res-lbl">{T['severity']}</div>
                    <div class="res-val">
                        <span class="sev-badge" style="background:{d['color']}18;color:{d['color']};border:1px solid {d['color']}35;">{d['severity']}</span>
                    </div>
                    <div class="res-lbl">{T['treatment']}</div>
                    <div class="res-val">{d['treatment']}</div>
                    <div class="res-lbl">{T['prevention']}</div>
                    <div class="res-val">{d['prevention']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown(f'<div class="probs-wrap"><div class="probs-title">{T["prob_title"]}</div>', unsafe_allow_html=True)
            for cls, prob in sorted(r["probs"].items(), key=lambda x: x[1], reverse=True):
                dc      = info[lang][cls]
                pct     = int(prob * 100)
                is_top  = cls == r["class"]
                clr     = dc["color"] if is_top else "rgba(255,255,255,0.12)"
                st.markdown(f"""
                <div class="pb-row">
                    <div class="pb-lbl">{dc['emoji']} {cls}</div>
                    <div class="pb-bg"><div class="pb-fill" style="width:{pct}%;background:{clr};"></div></div>
                    <div class="pb-pct">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button(f"🔄 {T['reset']}", key="reset"):
                st.session_state.result = None
                st.rerun()

        else:
            # Info modèle
            cls_list = " · ".join([info[lang][c]["name"] for c in info["classes"]])
            st.markdown(f"""
            <div class="info-card">
                <div style="font-size:9px;text-transform:uppercase;letter-spacing:2px;color:rgba(255,255,255,0.2);margin-bottom:22px;">{T['model_info']}</div>
                <div class="info-row">
                    <div class="info-lbl">{T['accuracy']}</div>
                    <div class="info-val" style="color:{info['color']};">{info['accuracy']}</div>
                </div>
                <div class="info-row">
                    <div class="info-lbl">{T['classes_lbl']}</div>
                    <div class="info-val-sm">{cls_list}</div>
                </div>
                <div class="info-row">
                    <div class="info-lbl">{T['size_lbl']}</div>
                    <div class="info-val-sm">{info['arch']}</div>
                </div>
                <div class="info-row">
                    <div class="info-lbl">Fichier</div>
                    <div class="info-val-sm" style="font-family:monospace;font-size:12px;color:rgba(255,255,255,0.35);">{info['model_file']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="how-section">
    <div class="section-label">{T['how_title']}</div>
    <div class="section-title">Simple. Rapide. Fiable.</div>
    <div class="steps">
        <div class="step">
            <div class="step-n">01</div>
            <div class="step-t">{T['step1']}</div>
            <div class="step-d">{T['step1_desc']}</div>
        </div>
        <div class="step" style="border-top-color:rgba(243,156,18,0.5);">
            <div class="step-n" style="color:rgba(243,156,18,0.1);">02</div>
            <div class="step-t">{T['step2']}</div>
            <div class="step-d">{T['step2_desc']}</div>
        </div>
        <div class="step" style="border-top-color:rgba(39,174,96,0.8);">
            <div class="step-n">03</div>
            <div class="step-t">{T['step3']}</div>
            <div class="step-d">{T['step3_desc']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-brand">🌿 AgriSmart</div>
    <div class="footer-copy">D-CLIC · OIF · 2025 · Lomé, Togo</div>
    <div class="footer-copy">CNN · Keras · TensorFlow · On-device AI</div>
</div>
""", unsafe_allow_html=True)