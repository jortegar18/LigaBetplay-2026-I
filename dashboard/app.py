"""
🏟️ Dashboard Liga BetPlay I-2026 — Data Science & ML
Datos REALES de la temporada actual + predicciones de tabla final.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.live_scraper import get_live_data as get_live_scraped_data, is_live_available
from src.data.real_data_2026 import (
    get_standings_df, get_results_df, get_upcoming_df, 
    TEAMS_2026, STANDINGS_2026
)
from src.models.elo_rating import EloSystem

# ─── Config ───
st.set_page_config(
    page_title="Liga BetPlay I-2026 | Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"

COLORS = {
    "gold": "#FCD116",
    "blue": "#003893",
    "red": "#CE1126",
    "green": "#2ECC71",
    "palette": ["#003893", "#CE1126", "#FCD116", "#2ECC71", "#9B59B6",
                "#E67E22", "#1ABC9C", "#E74C3C", "#3498DB", "#F39C12"]
}


# ─── Custom CSS ───
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #003893 0%, #1a1a2e 100%);
                border-radius: 12px; padding: 16px; border: 1px solid #003893; }
    .stMetric label { color: #FCD116 !important; font-weight: bold; }
    h1 { color: #FCD116; text-align: center; }
    h2, h3 { color: #3498DB; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #003893 0%, #0e1117 100%); }
    div[data-testid="stSidebar"] .stMarkdown { color: white; }
    .highlight-row { background-color: rgba(252, 209, 22, 0.1); }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ───
@st.cache_data(ttl=3600)
def load_live_data():
    """Carga datos en vivo (scraping FBref) con fallback a datos locales."""
    try:
        return get_live_scraped_data()
    except Exception:
        return get_standings_df(), get_results_df(), get_upcoming_df()

@st.cache_data
def load_predictions():
    path = PROCESSED_DIR / "season_predictions.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    return None

@st.cache_data
def load_heatmap():
    path = PROCESSED_DIR / "position_heatmap.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    return None

standings, df, upcoming_data = load_live_data()
teams = standings["team"].tolist() if not standings.empty and "team" in standings.columns else TEAMS_2026

# Detectar fecha actual
current_fecha = int(df["fecha"].max()) if not df.empty and "fecha" in df.columns else 7


# ─── Sidebar ───
st.sidebar.markdown("# ⚽ Liga BetPlay")
st.sidebar.markdown("### 🇨🇴 Temporada I-2026")

# Datos live con FBref
cache_file = BASE_DIR / "data" / "cache" / "live_standings.json"
if cache_file.exists():
    st.sidebar.success(f"📡 Datos en vivo (Fecha {current_fecha})")
else:
    st.sidebar.warning(f"📋 Datos locales (Fecha {current_fecha})")

if st.sidebar.button("🔄 Actualizar datos"):
    # Borrar caché
    import shutil
    cache_dir = BASE_DIR / "data" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["🏆 Posiciones", "📊 Resultados", "🔮 Predicciones", "⚡ Predictor"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("*Proyecto Data Science & ML*")


# ════════════════════════════════════════════
# PÁGINA 1: TABLA DE POSICIONES ACTUAL
# ════════════════════════════════════════════
if page == "🏆 Posiciones":
    st.markdown("# 🏆 Liga BetPlay I-2026 — Tabla de Posiciones")
    st.markdown("##### Fecha 7 completada — 20 de febrero de 2026")
    st.markdown("---")
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Partidos Jugados", f"{len(df)}")
    k2.metric("Goles Totales", f"{df['total_goals'].sum()}")
    k3.metric("Promedio Goles", f"{df['total_goals'].mean():.2f}")
    home_win_pct = (df["result"] == "H").mean() * 100
    k4.metric("Ventaja Local", f"{home_win_pct:.1f}%")
    
    st.markdown("---")
    
    # Tabla de posiciones con formato
    st.markdown("### 📋 Clasificación Actual")
    
    # Crear tabla con colores
    display_df = standings.copy()
    display_df.columns = ["Equipo", "PJ", "G", "E", "P", "GF", "GC", "DIF", "PTS"]
    
    # Clasificados a cuadrangulares (top 8)
    st.markdown("> 🟢 **Top 8** → Clasifican a cuadrangulares semifinales")
    
    st.dataframe(
        display_df.style.apply(
            lambda x: ["background-color: rgba(46, 204, 113, 0.15)" if x.name <= 8 
                       else "background-color: rgba(231, 76, 60, 0.15)" if x.name >= 17
                       else "" for _ in x], axis=1
        ).format({"PTS": "{:.0f}", "DIF": "{:+.0f}"}),
        use_container_width=True,
        height=740
    )
    
    # Gráfico de puntos
    st.markdown("### 📊 Puntos por Equipo")
    fig_pts = px.bar(
        display_df.reset_index(), x="PTS", y="Equipo", orientation="h",
        color="PTS", color_continuous_scale=["#CE1126", "#FCD116", "#003893"],
        text="PTS"
    )
    fig_pts.update_layout(
        template="plotly_dark", height=600, yaxis=dict(autorange="reversed"),
        showlegend=False, coloraxis_showscale=False
    )
    fig_pts.update_traces(textposition="outside")
    st.plotly_chart(fig_pts, use_container_width=True)


# ════════════════════════════════════════════
# PÁGINA 2: RESULTADOS POR FECHA
# ════════════════════════════════════════════
elif page == "📊 Resultados":
    st.markdown("# 📊 Resultados por Fecha")
    st.markdown("---")
    
    # Selector de fecha
    fecha = st.selectbox("Selecciona la fecha", range(1, 8), index=6, 
                          format_func=lambda x: f"Fecha {x}")
    
    fecha_data = df[df["fecha"] == fecha]
    
    # Mostrar partidos
    for _, match in fecha_data.iterrows():
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])
        
        with col1:
            st.markdown(f"**{match['home_team']}**")
        with col2:
            st.markdown(f"### {int(match['home_goals'])}")
        with col3:
            st.markdown("### —")
        with col4:
            st.markdown(f"### {int(match['away_goals'])}")
        with col5:
            st.markdown(f"**{match['away_team']}**")
        
        st.markdown("---")
    
    # Estadísticas de la fecha
    st.markdown(f"### 📈 Estadísticas Fecha {fecha}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Partidos", f"{len(fecha_data)}")
    c2.metric("Goles", f"{fecha_data['total_goals'].sum():.0f}")
    c3.metric("Promedio", f"{fecha_data['total_goals'].mean():.2f}")
    
    # Distribución de resultados
    st.markdown("### Distribución de Resultados (Todas las Fechas)")
    results_count = df["result"].value_counts()
    labels = {"H": f"Local ({results_count.get('H',0)})", 
              "D": f"Empate ({results_count.get('D',0)})", 
              "A": f"Visitante ({results_count.get('A',0)})"}
    
    fig_results = px.pie(
        values=[results_count.get("H", 0), results_count.get("D", 0), results_count.get("A", 0)],
        names=[labels["H"], labels["D"], labels["A"]],
        color_discrete_sequence=[COLORS["blue"], COLORS["gold"], COLORS["red"]],
        hole=0.4
    )
    fig_results.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_results, use_container_width=True)
    
    # Próxima fecha
    st.markdown("### 📅 Próxima Fecha (Fecha 8)")
    upcoming = get_upcoming_df()
    for _, match in upcoming.iterrows():
        col1, col2, col3 = st.columns([4, 2, 4])
        with col1:
            st.markdown(f"**{match['home_team']}**")
        with col2:
            st.markdown(f"*{match['date'].strftime('%d/%m')}*")
        with col3:
            st.markdown(f"**{match['away_team']}**")
        st.markdown("---")


# ════════════════════════════════════════════
# PÁGINA 3: PREDICCIONES DE TABLA FINAL
# ════════════════════════════════════════════
elif page == "🔮 Predicciones":
    st.markdown("# 🔮 Predicción de Tabla Final")
    st.markdown("##### Simulación Monte Carlo basada en ELO ratings")
    st.markdown("---")
    
    predictions = load_predictions()
    
    if predictions is None:
        st.warning("⚠️ No se han generado predicciones aún.")
        
        n_sims = st.slider("Número de simulaciones", 100, 2000, 500, 100)
        
        if st.button("🎲 Generar Predicciones", type="primary", use_container_width=True):
            from src.models.season_predictor import predict_season
            with st.spinner(f"Simulando {n_sims} temporadas..."):
                predictions = predict_season(n_simulations=n_sims)
            st.success("✅ Predicciones generadas!")
            st.cache_data.clear()
            st.rerun()
    
    if predictions is not None:
        # KPIs
        top1 = predictions.loc[predictions["prob_top1"].idxmax()]
        
        k1, k2, k3 = st.columns(3)
        k1.metric("🥇 Favorito al Título", top1["team"], 
                   f"{top1['prob_top1']:.1%} probabilidad")
        k2.metric("📊 Pts Promedio Líder", f"{predictions['avg_final_pts'].max():.1f}")
        k3.metric("Simulaciones", f"{len(predictions)}")
        
        st.markdown("---")
        
        # Tabla de predicciones
        st.markdown("### 📋 Predicción de Clasificación Final")
        
        pred_display = predictions[[
            "team", "current_pts", "avg_final_pts", "avg_pos",
            "prob_top1", "prob_top4", "prob_top8", "prob_bottom4"
        ]].copy()
        pred_display.columns = [
            "Equipo", "Pts Actual", "Pts Final (Prom)", "Pos Promedio",
            "% Campeón", "% Top 4", "% Top 8 (Clasifica)", "% Último 4"
        ]
        
        # Formatear porcentajes
        for col in ["% Campeón", "% Top 4", "% Top 8 (Clasifica)", "% Último 4"]:
            pred_display[col] = (pred_display[col] * 100).round(1).astype(str) + "%"
        
        pred_display["Pts Final (Prom)"] = pred_display["Pts Final (Prom)"].round(1)
        pred_display["Pos Promedio"] = pred_display["Pos Promedio"].round(1)
        
        st.dataframe(pred_display, use_container_width=True, height=740)
        
        # Gráfico: Probabilidad de clasificar (Top 8)
        st.markdown("### 🎯 Probabilidad de Clasificación (Top 8)")
        
        prob_data = predictions.sort_values("prob_top8", ascending=True)
        fig_prob = px.bar(
            prob_data, x="prob_top8", y="team", orientation="h",
            text=prob_data["prob_top8"].apply(lambda x: f"{x:.0%}"),
            color="prob_top8",
            color_continuous_scale=["#CE1126", "#FCD116", "#2ECC71"]
        )
        fig_prob.update_layout(
            template="plotly_dark", height=600,
            xaxis_tickformat=".0%", xaxis_title="Probabilidad",
            yaxis_title="", coloraxis_showscale=False
        )
        fig_prob.update_traces(textposition="outside")
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # Gráfico: Rango de puntos final
        st.markdown("### 📊 Rango de Puntos Esperado")
        
        range_data = predictions.sort_values("avg_final_pts", ascending=False)
        fig_range = go.Figure()
        fig_range.add_trace(go.Bar(
            x=range_data["team"], y=range_data["avg_final_pts"],
            name="Promedio", marker_color=COLORS["blue"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=range_data["max_pts"] - range_data["avg_final_pts"],
                arrayminus=range_data["avg_final_pts"] - range_data["min_pts"],
                color=COLORS["gold"]
            )
        ))
        fig_range.update_layout(
            template="plotly_dark", height=500,
            yaxis_title="Puntos", xaxis_title="",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_range, use_container_width=True)
        
        # ── MAPA DE CALOR DE POSICIONES ──
        st.markdown("---")
        st.markdown("### 🔥 Mapa de Calor — Probabilidad por Posición")
        st.markdown("*Cada celda muestra la probabilidad (%) de que un equipo termine en esa posición.*")
        
        heatmap_data = load_heatmap()
        
        if heatmap_data is not None:
            # Convertir a porcentajes para display
            z_values = (heatmap_data.values * 100).round(1)
            teams_ordered = heatmap_data.index.tolist()
            positions = list(range(1, len(heatmap_data.columns) + 1))
            
            # Texto para cada celda
            text_vals = [[f"{v:.1f}%" if v >= 1.0 else (f"{v:.1f}" if v > 0 else "") 
                          for v in row] for row in z_values]
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=z_values,
                x=[str(p) for p in positions],
                y=teams_ordered,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 9},
                colorscale=[
                    [0.0, "#1a1a2e"],
                    [0.05, "#16213e"],
                    [0.15, "#0f3460"],
                    [0.3, "#e94560"],
                    [0.5, "#FCD116"],
                    [1.0, "#2ECC71"]
                ],
                colorbar=dict(title="Prob %", ticksuffix="%"),
                hovertemplate="<b>%{y}</b><br>Posición %{x}<br>Probabilidad: %{z:.1f}%<extra></extra>"
            ))
            
            fig_heat.update_layout(
                template="plotly_dark",
                height=700,
                xaxis_title="Posición Final",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(dtick=1),
                margin=dict(l=150)
            )
            
            # Línea divisoria para Top 8
            fig_heat.add_vline(x=7.5, line_dash="dash", line_color=COLORS["gold"], 
                               line_width=2, annotation_text="Top 8", 
                               annotation_position="top")
            
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("El mapa de calor se generará al ejecutar las predicciones.")


# ════════════════════════════════════════════
# PÁGINA 4: PREDICTOR DE PARTIDO
# ════════════════════════════════════════════
elif page == "⚡ Predictor":
    st.markdown("# ⚡ Predictor de Resultado")
    st.markdown("##### Basado en ELO ratings de la Liga BetPlay I-2026")
    st.markdown("---")
    
    # Calcular ELO con datos reales
    @st.cache_resource
    def get_elo():
        elo = EloSystem(initial_elo=1500, k_factor=40, home_advantage=70)
        elo.process_matches(df)
        return elo
    
    elo = get_elo()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏠 Equipo Local")
        home_team = st.selectbox("Selecciona equipo local", teams, key="home")
    with col2:
        st.markdown("### ✈️ Equipo Visitante")
        away_team = st.selectbox("Selecciona equipo visitante", 
                                  [t for t in teams if t != home_team], key="away")
    
    if st.button("⚽ Predecir Resultado", type="primary", use_container_width=True):
        prediction = elo.predict_match(home_team, away_team)
        
        st.markdown("---")
        
        # Probabilidades
        p1, p2, p3 = st.columns(3)
        p1.metric(f"🏠 {home_team}", f"{prediction['home_win']:.1%}")
        p2.metric("🤝 Empate", f"{prediction['draw']:.1%}")
        p3.metric(f"✈️ {away_team}", f"{prediction['away_win']:.1%}")
        
        # Barra visual
        fig_bar = go.Figure(go.Bar(
            x=[prediction['home_win'], prediction['draw'], prediction['away_win']],
            y=[home_team, "Empate", away_team],
            orientation="h",
            marker_color=[COLORS["blue"], COLORS["gold"], COLORS["red"]],
            text=[f"{prediction['home_win']:.1%}", f"{prediction['draw']:.1%}", 
                  f"{prediction['away_win']:.1%}"],
            textposition="auto"
        ))
        fig_bar.update_layout(template="plotly_dark", height=200, 
                               xaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # ELO
        st.info(f"**ELO Rating**: {home_team} ({prediction['home_elo']}) vs "
                f"{away_team} ({prediction['away_elo']})")
        
        # Head-to-head
        st.markdown("### 📜 Enfrentamientos esta Temporada")
        h2h = df[
            ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
            ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        ].sort_values("date", ascending=False)
        
        if not h2h.empty:
            for _, match in h2h.iterrows():
                result_emoji = "🏠" if match["result"] == "H" else ("🤝" if match["result"] == "D" else "✈️")
                st.markdown(
                    f"{result_emoji} **{match['home_team']}** {int(match['home_goals'])} — "
                    f"{int(match['away_goals'])} **{match['away_team']}** "
                    f"*(Fecha {int(match['fecha'])})*"
                )
        else:
            st.info("No se han enfrentado esta temporada aún.")
    
    # Ranking ELO
    st.markdown("---")
    st.markdown("### 🏅 Ranking ELO Actual")
    rankings = elo.get_rankings()
    st.dataframe(rankings, use_container_width=True, height=500)
