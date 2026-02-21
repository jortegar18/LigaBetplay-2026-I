"""
team_clustering.py — Clustering de equipos por estilo de juego Liga BetPlay
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def compute_team_profiles(df: pd.DataFrame, season: int = None) -> pd.DataFrame:
    """
    Calcula el perfil estadístico de cada equipo para clustering.
    
    Features por equipo:
    - Promedio goles a favor y en contra por partido
    - % victorias local y visitante
    - Promedio gol diferencia
    - % empates
    - Goles en últimos 15 min (si disponible)
    """
    data = df.copy()
    if season is not None:
        data = data[data["season"] == season]
    
    teams = sorted(set(data["home_team"].unique()) | set(data["away_team"].unique()))
    profiles = []
    
    for team in teams:
        home = data[data["home_team"] == team]
        away = data[data["away_team"] == team]
        
        total_played = len(home) + len(away)
        if total_played == 0:
            continue
        
        # Goles
        gf = home["home_goals"].sum() + away["away_goals"].sum()
        ga = home["away_goals"].sum() + away["home_goals"].sum()
        
        # Victorias
        home_wins = (home["result"] == "H").sum()
        away_wins = (away["result"] == "A").sum()
        draws = (home["result"] == "D").sum() + (away["result"] == "D").sum()
        
        profiles.append({
            "team": team,
            "matches": total_played,
            "avg_gf": gf / total_played,
            "avg_ga": ga / total_played,
            "avg_gd": (gf - ga) / total_played,
            "win_rate": (home_wins + away_wins) / total_played,
            "draw_rate": draws / total_played,
            "loss_rate": 1 - (home_wins + away_wins + draws) / total_played,
            "home_win_rate": home_wins / max(len(home), 1),
            "away_win_rate": away_wins / max(len(away), 1),
            "home_strength": (home_wins * 3 + (home["result"] == "D").sum()) / max(len(home) * 3, 1),
            "away_strength": (away_wins * 3 + (away["result"] == "D").sum()) / max(len(away) * 3, 1),
            "total_gf": gf,
            "total_ga": ga,
        })
    
    return pd.DataFrame(profiles)


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 10) -> int:
    """Encuentra el número óptimo de clusters usando silhouette score."""
    scores = []
    k_range = range(2, min(max_k + 1, len(X_scaled)))
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))
    
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"Óptimo k={best_k} (silhouette={max(scores, key=lambda x: x[1])[1]:.3f})")
    
    return best_k


def cluster_teams(df: pd.DataFrame, season: int = None, 
                   n_clusters: int = None) -> tuple:
    """
    Agrupa equipos por estilo de juego.
    
    Returns:
        (profiles_df con cluster asignado, pca_df para visualización, kmeans, scaler, pca)
    """
    profiles = compute_team_profiles(df, season)
    
    feature_cols = ["avg_gf", "avg_ga", "avg_gd", "win_rate", "draw_rate",
                     "home_win_rate", "away_win_rate", "home_strength", "away_strength"]
    
    X = profiles[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encontrar k óptimo si no se especifica
    if n_clusters is None:
        n_clusters = find_optimal_k(X_scaled)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles["cluster"] = kmeans.fit_predict(X_scaled)
    
    # PCA para visualización 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    profiles["pca_x"] = coords[:, 0]
    profiles["pca_y"] = coords[:, 1]
    
    # Nombres descriptivos para clusters
    cluster_names = {}
    for c in range(n_clusters):
        cluster_teams = profiles[profiles["cluster"] == c]
        avg_gf = cluster_teams["avg_gf"].mean()
        avg_ga = cluster_teams["avg_ga"].mean()
        win_rate = cluster_teams["win_rate"].mean()
        
        if win_rate > 0.45 and avg_gf > 1.3:
            name = "⚔️ Ofensivo Dominante"
        elif win_rate > 0.35 and avg_ga < 1.0:
            name = "🛡️ Defensivo Sólido"
        elif avg_gf > 1.2 and avg_ga > 1.2:
            name = "🎭 Alto Riesgo"
        elif win_rate < 0.3:
            name = "📉 En Dificultad"
        else:
            name = "⚖️ Equilibrado"
        
        cluster_names[c] = name
    
    profiles["cluster_name"] = profiles["cluster"].map(cluster_names)
    
    # Guardar
    output_path = PROCESSED_DIR / "team_clusters.csv"
    profiles.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    joblib.dump(kmeans, MODELS_DIR / "team_clustering.pkl")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.pkl")
    
    print(f"\nClusters guardados: {output_path}")
    
    return profiles, pca, kmeans, scaler


if __name__ == "__main__":
    print("=" * 60)
    print("  Team Clustering — Liga BetPlay")
    print("=" * 60)
    
    data_path = PROCESSED_DIR / "matches_clean.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", encoding="utf-8-sig")
    
    # Clustering con datos más recientes
    profiles, pca, kmeans, scaler = cluster_teams(df, season=2024)
    
    print("\n📊 Perfiles por cluster:")
    for cluster_name in profiles["cluster_name"].unique():
        cluster_data = profiles[profiles["cluster_name"] == cluster_name]
        teams = ", ".join(cluster_data["team"].values[:5])
        print(f"\n  {cluster_name}:")
        print(f"    Equipos: {teams}")
        print(f"    Avg GF: {cluster_data['avg_gf'].mean():.2f}")
        print(f"    Avg GA: {cluster_data['avg_ga'].mean():.2f}")
        print(f"    Win Rate: {cluster_data['win_rate'].mean():.1%}")
