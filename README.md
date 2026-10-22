# AgriSmart (Streamlit) : diagnostic des maladies du maïs et du manioc

Application Streamlit qui reconnaît la maladie d'une feuille de maïs ou de manioc sur une photo, puis donne sa gravité, un traitement et des mesures de prévention. L'interface est disponible en français et en anglais.

## Fonctionnement

1. L'utilisateur choisit la culture et envoie une photo (JPG ou PNG).
2. L'image est redimensionnée en 160 × 160 pixels et passée au réseau de neurones convolutif de la culture.
3. L'application affiche la classe la plus probable, les probabilités de chaque classe, puis la fiche de la maladie : gravité, traitement, prévention.

| Culture | Classes | Modèle |
|---|---|---|
| Maïs | feuille saine, brûlure (blight), rouille, cercosporiose | `cnn_simple_best.keras` |
| Manioc | feuille saine, mosaïque (CMD), bactériose (CBB), acarien vert (CGM), striure brune (CBSD) | `cnn_best.keras` |

## Installation

Prérequis : Python 3.11 (voir `runtime.txt`).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Projet lié

[app_agrismart](https://github.com/pierredamien21/app_agrismart) reprend les mêmes modèles dans une architecture web séparée : API FastAPI et interface React.

## Stack

Python · Streamlit · TensorFlow / Keras · NumPy · Pillow

## Limites

- Les performances des modèles (précision par classe, jeu de test) ne sont pas documentées dans ce dépôt.
- Les traitements proposés sont des conseils généraux : ils ne remplacent pas l'avis d'un agronome.
