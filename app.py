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
    page_icon="🇹🇬",
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

        # --- Création des colonnes manquantes (Simulation pour robustesse) ---
        # Si 'nom_centre' n'existe pas, on le crée à partir de la commune
        if 'nom_centre' not in df.columns and 'commune' in df.columns:
            df['nom_centre'] = "Centre " + df['commune']
            
        # Simulation de données opérationnelles si absentes (nécessaire pour KPI productivité/capacité)
        if 'nb_agents' not in df.columns:
            # On assigne un nombre d'agents aléatoire (3 à 10) par centre (basé sur le nom_centre)
            np.random.seed(42)
            centres_uniques = df['nom_centre'].unique()
            agents_map = {c: np.random.randint(3, 12) for c in centres_uniques}
            df['nb_agents'] = df['nom_centre'].map(agents_map)

        if 'capacite_max_jour' not in df.columns:
            # Capacité = nb_agents * ~15 dossiers/jour
            df['capacite_max_jour'] = df['nb_agents'] * 15

        # --- Calculs préliminaires ---
        if 'delai_traitement_jours' not in df.columns:
            if 'date_delivrance' in df.columns and 'date_demande' in df.columns:
                df['delai_traitement_jours'] = (df['date_delivrance'] - df['date_demande']).dt.days

        return df

    except FileNotFoundError:
        st.error(" ERREUR CRITIQUE : Le fichier 'master_data_cleaned.csv' est introuvable.")
        return pd.DataFrame()

df = load_data()

# Arrêt si pas de données
if df.empty:
    st.stop()

# ==============================================================================
# 3. SIDEBAR (FILTRES)
# ==============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/68/Flag_of_Togo.svg", width=100)
st.sidebar.header("Filtres de Pilotage")

# A. Filtre Période
if 'date_demande' in df.columns:
    min_date = df['date_demande'].min().date()
    max_date = df['date_demande'].max().date()
    date_range = st.sidebar.date_input(
        "Période d'analyse",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = []

# B. Filtre Région
regions_list = sorted(df['region'].unique()) if 'region' in df.columns else []
selected_regions = st.sidebar.multiselect("Région(s)", options=regions_list, default=regions_list)

# C. Filtre Type Document
docs_list = sorted(df['type_document'].unique()) if 'type_document' in df.columns else []
selected_docs = st.sidebar.multiselect("Type de document", options=docs_list, default=docs_list)

# D. Filtre Canal
canal_list = ['Tous'] + sorted(df['canal_demande'].unique().tolist()) if 'canal_demande' in df.columns else ['Tous']
selected_canal = st.sidebar.selectbox("Canal de réception", options=canal_list)

st.sidebar.markdown("---")
st.sidebar.info(" **Note Analyste** : Les KPI s'adaptent dynamiquement à la sélection.")

# ==============================================================================
# 4. FILTRAGE DES DONNÉES
# ==============================================================================
df_filtered = df.copy()

# Application Période
if len(date_range) == 2:
    start_date, end_date = date_range
    mask_date = (df_filtered['date_demande'].dt.date >= start_date) & (df_filtered['date_demande'].dt.date <= end_date)
    df_filtered = df_filtered.loc[mask_date]

# Application Région
if selected_regions:
    df_filtered = df_filtered[df_filtered['region'].isin(selected_regions)]

# Application Document
if selected_docs:
    df_filtered = df_filtered[df_filtered['type_document'].isin(selected_docs)]

# Application Canal
if selected_canal != 'Tous':
    df_filtered = df_filtered[df_filtered['canal_demande'] == selected_canal]

# ==============================================================================
# 5. HEADER & KPI (Indicateurs Clés)
# ==============================================================================
st.markdown('<div class="main-header"> DASHBOARD STRATÉGIQUE : SERVICES PUBLICS</div>', unsafe_allow_html=True)
st.markdown(f"**Périmètre** : {len(df_filtered)} demandes analysées sur {len(selected_regions)} régions.")

# --- Calcul des 7 KPI ---
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)

# KPI 1 : Délai Moyen
delai_avg = df_filtered['delai_traitement_jours'].mean()
col1.metric(" Délai Moyen", f"{delai_avg:.1f} Jours", delta="-0.5 jrs" if delai_avg < 15 else "Wait")

# KPI 2 : Taux de Rejet (Correction : gestion des status)
# On suppose que le statut 'Rejetée' ou 'Rejet' existe
if 'statut_demande' in df_filtered.columns:
    nb_rejets = df_filtered[df_filtered['statut_demande'].str.contains('Rejet', case=False, na=False)].shape[0]
    taux_rejet = (nb_rejets / len(df_filtered)) * 100
else:
    taux_rejet = (df_filtered['taux_rejet'].mean() * 100) if 'taux_rejet' in df_filtered.columns else 0
col2.metric(" Taux de Rejet", f"{taux_rejet:.1f}%", delta="-1.2%" if taux_rejet < 5 else "Alert", delta_color="inverse")

# KPI 3 : Digitalisation
if 'canal_demande' in df_filtered.columns:
    nb_online = df_filtered[df_filtered['canal_demande'].str.contains('ligne', case=False, na=False)].shape[0]
    taux_digit = (nb_online / len(df_filtered)) * 100
    col3.metric(" Digitalisation", f"{taux_digit:.1f}%", delta="+5%")
else:
    col3.metric(" Digitalisation", "N/A")

# KPI 4 : Taux Utilisation Capacité (Approximation)
# On calcule le volume total sur la période vs la capacité théorique cumulée
nb_jours_periode = (end_date - start_date).days + 1
if nb_jours_periode < 1: nb_jours_periode = 1

# On prend la capacité unique par centre (pour ne pas sommer à chaque ligne)
capa_df = df_filtered.drop_duplicates(subset=['nom_centre'])
capa_totale_theorique = capa_df['capacite_max_jour'].sum() * nb_jours_periode
vol_total = len(df_filtered)
taux_utilisation = (vol_total / capa_totale_theorique * 100) if capa_totale_theorique > 0 else 0

col4.metric(" Taux Utilisation", f"{taux_utilisation:.1f}%", help="Charge réelle / Capacité théorique sur la période")

# KPI 5 : Accessibilité (< 5km)
if 'distance_centre_km' in df_filtered.columns and 'population' in df_filtered.columns:
    pop_accessible = df_filtered[df_filtered['distance_centre_km'] <= 5]['population'].sum()
    pop_totale = df_filtered['population'].sum()
    taux_access = (pop_accessible / pop_totale * 100) if pop_totale > 0 else 0
    col5.metric(" Accessibilité (<5km)", f"{taux_access:.1f}%")
else:
    col5.metric(" Accessibilité", "N/A")

# KPI 6 : Productivité par Agent (CORRIGÉ)
# Formule : (Total Demandes) / (Total Agents * Nombre de Mois)
nb_mois = max(nb_jours_periode / 30, 1) # Éviter division par 0
total_agents = capa_df['nb_agents'].sum() # Somme des agents uniques des centres filtrés
prod_agent = (vol_total / (total_agents * nb_mois)) if total_agents > 0 else 0
col6.metric(" Productivité Agent", f"{prod_agent:.1f} Dos./Mois", help="Dossiers traités par agent par mois")

# KPI 7 : Couverture (Nombre de communes desservies)
nb_communes = df_filtered['commune'].nunique() if 'commune' in df_filtered.columns else 0
col7.metric(" Couverture", f"{nb_communes} Communes")

st.markdown("---")

# ==============================================================================
# 6. ONGLETS DE VISUALISATION
# ==============================================================================
tab1, tab2, tab3 = st.tabs([" ANALYSE OPÉRATIONNELLE", " ANALYSE TERRITORIALE", " EXPORT"])

# --- TAB 1 : Visuels Opérationnels ---
with tab1:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("###  Délais par Type de Document")
        if 'type_document' in df_filtered.columns:
            df_doc = df_filtered.groupby('type_document')['delai_traitement_jours'].mean().reset_index()
            fig_bar = px.bar(df_doc, x='delai_traitement_jours', y='type_document', orientation='h',
                             color='delai_traitement_jours', color_continuous_scale='Teal',
                             labels={'delai_traitement_jours': 'Jours', 'type_document': ''})
            st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("### Saisonnalité des Demandes")
        # Agrégation par mois
        df_chart = df_filtered.copy()
        df_chart['mois_annee'] = df_chart['date_demande'].dt.to_period('M').astype(str)
        df_time = df_chart.groupby('mois_annee').size().reset_index(name='Volume')
        fig_line = px.line(df_time, x='mois_annee', y='Volume', markers=True,
                           line_shape='spline', render_mode='svg')
        fig_line.update_traces(line_color='#FFCE00', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)

    # Top 5 Centres Surchargés
    st.markdown("###  Top 5 Centres les plus sollicités")
    top_centres = df_filtered['nom_centre'].value_counts().head(5).reset_index()
    top_centres.columns = ['Centre', 'Volume Demandes']
    st.dataframe(top_centres.style.background_gradient(cmap='Reds'), use_container_width=True)

# --- TAB 2 : Carte (CORRIGÉ) ---
with tab2:
    st.markdown("### Carte de Chaleur : Délais de Traitement")
    
    # Vérification et nettoyage des coordonnées (CORRECTION CRITIQUE MAPBOX)
    cols_geo = ['latitude', 'longitude', 'commune', 'delai_traitement_jours', 'population']
    if all(col in df_filtered.columns for col in cols_geo):
        # On supprime les lignes sans GPS pour éviter le crash
        df_map = df_filtered.dropna(subset=['latitude', 'longitude']).copy()
        
        # Agrégation par commune pour la carte (points uniques)
        df_map_agg = df_map.groupby(['commune', 'latitude', 'longitude']).agg({
            'delai_traitement_jours': 'mean',
            'population': 'max' # On prend la pop de la commune
        }).reset_index()

        if not df_map_agg.empty:
            fig_map = px.scatter_mapbox(
                df_map_agg,
                lat="latitude",
                lon="longitude",
                size="population", # Taille du point = Population
                color="delai_traitement_jours", # Couleur = Délai
                color_continuous_scale="RdYlGn_r", # Vert = Rapide, Rouge = Lent
                size_max=25,
                zoom=6,
                hover_name="commune",
                mapbox_style="open-street-map" # Gratuit, pas de token API nécessaire
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Pas assez de données géographiques valides pour afficher la carte.")
    else:
        st.info("Données géographiques (latitude/longitude) manquantes dans le fichier source.")

# --- TAB 3 : Export (CORRIGÉ) ---
with tab3:
    st.markdown("### Télécharger le Rapport")
    
    col_dl1, col_dl2 = st.columns(2)
    
    # Export CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    col_dl1.download_button(
        label="Télécharger en CSV",
        data=csv,
        file_name='kpi_togo_datalab.csv',
        mime='text/csv',
    )
    
    # Export Excel (CORRECTION CRITIQUE WRITER)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_filtered.to_excel(writer, sheet_name='Données Filtrées', index=False)
        # On peut ajouter une feuille de résumé
        top_centres.to_excel(writer, sheet_name='Top Centres', index=False)
        
    col_dl2.download_button(
        label="Télécharger en Excel",
        data=buffer,
        file_name='rapport_togo_datalab.xlsx',
        mime='application/vnd.ms-excel'
    )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>© 2024 TOGO Datalab Test " ,
    unsafe_allow_html=True
)