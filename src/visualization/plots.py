"""
plots.py — Visualizaciones para Liga BetPlay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"

# Colores de la bandera colombiana + complementarios
COLORS = {
    "gold": "#FCD116",
    "blue": "#003893",
    "red": "#CE1126",
    "dark": "#1a1a2e",
    "palette": ["#003893", "#CE1126", "#FCD116", "#2ECC71", "#9B59B6",
                "#E67E22", "#1ABC9C", "#E74C3C", "#3498DB", "#F39C12"]
}

sns.set_theme(style="darkgrid", palette=COLORS["palette"])
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12


def plot_goals_per_season(df: pd.DataFrame, save: bool = True):
    """Gráfico de goles promedio por temporada."""
    goals = df.groupby("season").agg(
        avg_total=("total_goals", "mean"),
        avg_home=("home_goals", "mean"),
        avg_away=("away_goals", "mean"),
    ).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=goals["season"], y=goals["avg_total"], 
                              name="Total", line=dict(width=3, color=COLORS["blue"])))
    fig.add_trace(go.Scatter(x=goals["season"], y=goals["avg_home"],
                              name="Local", line=dict(width=2, dash="dash", color=COLORS["gold"])))
    fig.add_trace(go.Scatter(x=goals["season"], y=goals["avg_away"],
                              name="Visitante", line=dict(width=2, dash="dot", color=COLORS["red"])))
    
    fig.update_layout(
        title="⚽ Promedio de Goles por Temporada — Liga BetPlay",
        xaxis_title="Temporada",
        yaxis_title="Goles promedio por partido",
        template="plotly_dark",
        hovermode="x unified"
    )
    
    if save:
        fig.write_html(str(REPORTS_DIR / "goals_per_season.html"))
    return fig


def plot_home_advantage(df: pd.DataFrame, save: bool = True):
    """Gráfico de ventaja local por temporada."""
    results = df.groupby("season")["result"].value_counts(normalize=True).unstack(fill_value=0)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=results.index, y=results.get("H", 0), name="Victoria Local",
                          marker_color=COLORS["blue"]))
    fig.add_trace(go.Bar(x=results.index, y=results.get("D", 0), name="Empate",
                          marker_color=COLORS["gold"]))
    fig.add_trace(go.Bar(x=results.index, y=results.get("A", 0), name="Victoria Visitante",
                          marker_color=COLORS["red"]))
    
    fig.update_layout(
        barmode="stack",
        title="🏠 Ventaja Local — Liga BetPlay",
        xaxis_title="Temporada",
        yaxis_title="Proporción",
        template="plotly_dark",
        yaxis_tickformat=".0%"
    )
    
    if save:
        fig.write_html(str(REPORTS_DIR / "home_advantage.html"))
    return fig


def plot_elo_evolution(elo_history: pd.DataFrame, teams: list = None,
                        save: bool = True):
    """Gráfico de evolución ELO de equipos seleccionados."""
    if teams is None:
        # Top 5 equipos por ELO final
        last = elo_history.groupby("home_team")["home_elo_after"].last()
        teams = last.nlargest(5).index.tolist()
    
    fig = go.Figure()
    
    for team in teams:
        team_data = elo_history[
            (elo_history["home_team"] == team) | (elo_history["away_team"] == team)
        ].copy()
        
        elos = []
        for _, row in team_data.iterrows():
            if row["home_team"] == team:
                elos.append({"date": row["date"], "elo": row["home_elo_after"]})
            else:
                elos.append({"date": row["date"], "elo": row["away_elo_after"]})
        
        elo_df = pd.DataFrame(elos)
        fig.add_trace(go.Scatter(x=elo_df["date"], y=elo_df["elo"], 
                                  name=team, mode="lines"))
    
    fig.add_hline(y=1500, line_dash="dash", line_color="gray", 
                   annotation_text="ELO base (1500)")
    
    fig.update_layout(
        title="📈 Evolución ELO — Liga BetPlay",
        xaxis_title="Fecha",
        yaxis_title="Rating ELO",
        template="plotly_dark",
        hovermode="x unified"
    )
    
    if save:
        fig.write_html(str(REPORTS_DIR / "elo_evolution.html"))
    return fig


def plot_team_clusters(profiles: pd.DataFrame, save: bool = True):
    """Scatter plot de clusters de equipos en 2D (PCA)."""
    fig = px.scatter(
        profiles, x="pca_x", y="pca_y",
        color="cluster_name",
        text="team",
        size="matches",
        hover_data=["avg_gf", "avg_ga", "win_rate"],
        title="🎯 Clustering de Equipos — Liga BetPlay",
        template="plotly_dark",
        color_discrete_sequence=COLORS["palette"]
    )
    
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
    )
    
    if save:
        fig.write_html(str(REPORTS_DIR / "team_clusters.html"))
    return fig


def plot_model_comparison(results: dict, save: bool = True):
    """Comparación de modelos ML."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    cv_means = [results[n]["cv_mean"] for n in names]
    
    fig = go.Figure(data=[
        go.Bar(name="Test Accuracy", x=names, y=accuracies, marker_color=COLORS["blue"]),
        go.Bar(name="CV Mean", x=names, y=cv_means, marker_color=COLORS["gold"]),
    ])
    
    fig.update_layout(
        barmode="group",
        title="🤖 Comparación de Modelos — Liga BetPlay",
        yaxis_title="Accuracy",
        template="plotly_dark",
        yaxis_range=[0, 1]
    )
    
    if save:
        fig.write_html(str(REPORTS_DIR / "model_comparison.html"))
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, 
                           model_name: str, save: bool = True):
    """Heatmap de matriz de confusión."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Matriz de Confusión — {model_name}", fontsize=14)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    plt.tight_layout()
    
    if save:
        plt.savefig(REPORTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png",
                     dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, model_name: str,
                              top_n: int = 15, save: bool = True):
    """Gráfico de importancia de features."""
    top = fi_df.head(top_n)
    
    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        title=f"📊 Feature Importance — {model_name}",
        template="plotly_dark",
        color="importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    
    if save:
        fig.write_html(str(REPORTS_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.html"))
    return fig
