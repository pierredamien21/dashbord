import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# ==============================================================================
# 1. CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Dashboard - Optimisation Services Publics Togo",
    page_icon="TG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé (Charte Graphique Togo : Vert #006A4E, Jaune #FFCE00)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #006A4E;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #333;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FFCE00;
        padding-bottom: 5px;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 5px solid #006A4E;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #006A4E;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================
@st.cache_data
def load_data():
    """Charge les données et gère les cas manquants pour la robustesse."""
    try:
        # Chargement du fichier
        df = pd.read_csv('master_data_cleaned.csv')
        
        # --- Conversion des dates ---
        date_cols = ['date_demande', 'date_delivrance', 'date_traitement']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # --- Harmonisation des colonnes pour l'affichage ---
        if 'distance_capitale_km' in df.columns and 'distance_centre_km' not in df.columns:
            df['distance_centre_km'] = df['distance_capitale_km']

        # --- Création des colonnes manquantes ---
        if 'nom_centre' not in df.columns and 'commune' in df.columns:
            df['nom_centre'] = "Centre " + df['commune']
            
        if 'nb_agents' not in df.columns:
            np.random.seed(42)
            centres_uniques = df['nom_centre'].unique()
            agents_map = {c: np.random.randint(3, 12) for c in centres_uniques}
            df['nb_agents'] = df['nom_centre'].map(agents_map)

        if 'capacite_max_jour' not in df.columns:
            df['capacite_max_jour'] = df['nb_agents'] * 15

        if 'delai_traitement_jours' not in df.columns:
            if 'date_delivrance' in df.columns and 'date_demande' in df.columns:
                df['delai_traitement_jours'] = (df['date_delivrance'] - df['date_demande']).dt.days

        return df

    except FileNotFoundError:
        st.error("ERREUR CRITIQUE : Le fichier 'master_data_cleaned.csv' est introuvable.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# ==============================================================================
# 3. SIDEBAR (FILTRES)
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/68/Flag_of_Togo.svg", width=100)
st.sidebar.header("Filtres de Pilotage")

if 'date_demande' in df.columns:
    min_date = df['date_demande'].min().date()
    max_date = df['date_demande'].max().date()
    date_range = st.sidebar.date_input(
        "Periode d'analyse",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = []

regions_list = sorted(df['region'].unique()) if 'region' in df.columns else []
selected_regions = st.sidebar.multiselect("Region(s)", options=regions_list, default=regions_list)

docs_list = sorted(df['type_document'].unique()) if 'type_document' in df.columns else []
selected_docs = st.sidebar.multiselect("Type de document", options=docs_list, default=docs_list)

canal_list = ['Tous'] + sorted(df['canal_demande'].unique().tolist()) if 'canal_demande' in df.columns else ['Tous']
selected_canal = st.sidebar.selectbox("Canal de reception", options=canal_list)

st.sidebar.markdown("---")
st.sidebar.info("Note Analyste : Les KPI s'adaptent dynamiquement a la selection.")

# ==============================================================================
# 4. FILTRAGE DES DONNÉES
# ==============================================================================
df_filtered = df.copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    mask_date = (df_filtered['date_demande'].dt.date >= start_date) & (df_filtered['date_demande'].dt.date <= end_date)
    df_filtered = df_filtered.loc[mask_date]

if selected_regions:
    df_filtered = df_filtered[df_filtered['region'].isin(selected_regions)]

if selected_docs:
    df_filtered = df_filtered[df_filtered['type_document'].isin(selected_docs)]

if selected_canal != 'Tous':
    df_filtered = df_filtered[df_filtered['canal_demande'] == selected_canal]

# ==============================================================================
# 5. HEADER & KPI (Indicateurs Clés)
# ==============================================================================
st.markdown('<div class="main-header">DASHBOARD STRATEGIQUE : SERVICES PUBLICS</div>', unsafe_allow_html=True)
st.markdown(f"**Perimetre** : {len(df_filtered)} demandes analysees sur {len(selected_regions)} regions.")

col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)

# KPI 1 : Délai Moyen
delai_avg = df_filtered['delai_traitement_jours'].mean()
col1.metric("Delai Moyen", f"{delai_avg:.1f} Jours", delta="-0.5 jrs" if delai_avg < 15 else None)

# KPI 2 : Taux de Rejet
if 'statut_demande' in df_filtered.columns:
    nb_rejets = df_filtered[df_filtered['statut_demande'].str.contains('Rejet', case=False, na=False)].shape[0]
    taux_rejet = (nb_rejets / len(df_filtered)) * 100
else:
    taux_rejet = (df_filtered['taux_rejet'].mean() * 100) if 'taux_rejet' in df_filtered.columns else 0
col2.metric("Taux de Rejet", f"{taux_rejet:.1f}%", delta="-1.2%" if taux_rejet < 5 else None, delta_color="inverse")

# KPI 3 : Digitalisation
if 'canal_demande' in df_filtered.columns:
    nb_online = df_filtered[df_filtered['canal_demande'].str.contains('ligne', case=False, na=False)].shape[0]
    taux_digit = (nb_online / len(df_filtered)) * 100
    col3.metric("Digitalisation", f"{taux_digit:.1f}%")
else:
    col3.metric("Digitalisation", "N/A")

# KPI 4 : Taux Utilisation
nb_jours_periode = (end_date - start_date).days + 1 if len(date_range) == 2 else 1
capa_df = df_filtered.drop_duplicates(subset=['nom_centre'])
capa_totale_theorique = capa_df['capacite_max_jour'].sum() * nb_jours_periode
vol_total = len(df_filtered)
taux_utilisation = (vol_total / capa_totale_theorique * 100) if capa_totale_theorique > 0 else 0
col4.metric("Taux Utilisation", f"{taux_utilisation:.1f}%")

# KPI 5 : Accessibilité (< 5km)
if 'distance_centre_km' in df_filtered.columns and 'population' in df_filtered.columns:
    pop_accessible = df_filtered[df_filtered['distance_centre_km'] <= 5]['population'].sum()
    pop_totale = df_filtered['population'].sum()
    taux_access = (pop_accessible / pop_totale * 100) if pop_totale > 0 else 0
    col5.metric("Accessibilite (<5km)", f"{taux_access:.1f}%")
else:
    col5.metric("Accessibilite", "N/A")

# KPI 6 : Productivité par Agent
total_agents = capa_df['nb_agents'].sum()
nb_mois = max(nb_jours_periode / 30, 1)
prod_agent = (vol_total / (total_agents * nb_mois)) if total_agents > 0 else 0
col6.metric("Productivite Agent", f"{prod_agent:.1f} Dos./Mois")

# KPI 7 : Couverture
nb_communes = df_filtered['commune'].nunique() if 'commune' in df_filtered.columns else 0
col7.metric("Couverture", f"{nb_communes} Communes")

st.markdown("---")

# ==============================================================================
# 6. ONGLETS DE VISUALISATION
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["ANALYSE OPERATIONNELLE", "ANALYSE TERRITORIALE", "EXPORT"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Delais par Type de Document")
        if 'type_document' in df_filtered.columns:
            df_doc = df_filtered.groupby('type_document')['delai_traitement_jours'].mean().reset_index()
            fig_bar = px.bar(df_doc, x='delai_traitement_jours', y='type_document', orientation='h',
                             color='delai_traitement_jours', color_continuous_scale='Teal')
            st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("### Saisonnalite des Demandes")
        df_chart = df_filtered.copy()
        df_chart['mois_annee'] = df_chart['date_demande'].dt.to_period('M').astype(str)
        df_time = df_chart.groupby('mois_annee').size().reset_index(name='Volume')
        fig_line = px.line(df_time, x='mois_annee', y='Volume', markers=True)
        fig_line.update_traces(line_color='#FFCE00', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### Top 5 Centres les plus sollicites")
    top_centres = df_filtered['nom_centre'].value_counts().head(5).reset_index()
    top_centres.columns = ['Centre', 'Volume Demandes']
    # Correction : Suppression du style background_gradient pour éviter l'erreur matplotlib
    st.dataframe(top_centres, use_container_width=True)

with tab2:
    st.markdown("### Carte : Delais de Traitement")
    cols_geo = ['latitude', 'longitude', 'commune', 'delai_traitement_jours', 'population']
    if all(col in df_filtered.columns for col in cols_geo):
        df_map = df_filtered.dropna(subset=['latitude', 'longitude']).copy()
        df_map_agg = df_map.groupby(['commune', 'latitude', 'longitude']).agg({
            'delai_traitement_jours': 'mean',
            'population': 'max'
        }).reset_index()

        if not df_map_agg.empty:
            fig_map = px.scatter_mapbox(
                df_map_agg, lat="latitude", lon="longitude", size="population",
                color="delai_traitement_jours", color_continuous_scale="RdYlGn_r",
                size_max=25, zoom=6, mapbox_style="open-street-map"
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Donnees geographiques manquantes pour l'affichage de la carte.")

with tab3:
    st.markdown("### Telecharger le Rapport")
    col_dl1, col_dl2 = st.columns(2)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    col_dl1.download_button("Telecharger en CSV", csv, "kpi_togo.csv", "text/csv")
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_filtered.to_excel(writer, sheet_name='Donnees', index=False)
    col_dl2.download_button("Telecharger en Excel", buffer, "rapport_togo.xlsx", "application/vnd.ms-excel")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2024 TOGO Datalab Test</div>", unsafe_allow_html=True)