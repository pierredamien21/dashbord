# Tableau de bord : délivrance des documents officiels au Togo

Tableau de bord Streamlit qui suit le traitement des demandes de documents officiels (carte d'identité, passeport, actes) au Togo en 2023. Il accompagne l'analyse du dépôt [togo-services-optimization](https://github.com/pierredamien21/togo-services-optimization).

**Démo en ligne :** https://dashbord-togo.streamlit.app (si l'application est en veille, cliquez sur le bouton pour la relancer ; elle redémarre en quelques secondes).

## Fonctionnalités

- **Filtres** : période, régions, type de document et canal de dépôt (en ligne ou guichet). Les indicateurs se recalculent à chaque changement.
- **Analyse opérationnelle** : délai moyen de traitement par type de document, évolution mensuelle du nombre de demandes.
- **Analyse territoriale** : carte des communes avec leurs indicateurs.
- **Export** : données filtrées en CSV et rapport Excel.

## Données

`master_data_cleaned.csv` est le jeu nettoyé produit par l'analyse : une ligne par demande, avec la commune, le type de document, le délai de traitement, le statut, le canal, des informations sur le demandeur et les coordonnées de la commune.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Python · Streamlit · pandas · Plotly · openpyxl · XlsxWriter
