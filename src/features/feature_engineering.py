"""
feature_engineering.py — Feature Engineering para predicción de partidos Liga BetPlay
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def calculate_form(df: pd.DataFrame, team: str, date: pd.Timestamp,
                   n_matches: int = 5) -> dict:
    """
    Calcula la forma reciente de un equipo (últimos N partidos).
    
    Returns:
        Dict con: wins, draws, losses, goals_for, goals_against, points, form_str
    """
    # Partidos del equipo antes de la fecha
    home_matches = df[(df["home_team"] == team) & (df["date"] < date)].copy()
    home_matches["team_goals"] = home_matches["home_goals"]
    home_matches["opp_goals"] = home_matches["away_goals"]
    home_matches["team_result"] = home_matches["result"].map({"H": "W", "D": "D", "A": "L"})
    
    away_matches = df[(df["away_team"] == team) & (df["date"] < date)].copy()
    away_matches["team_goals"] = away_matches["away_goals"]
    away_matches["opp_goals"] = away_matches["home_goals"]
    away_matches["team_result"] = away_matches["result"].map({"H": "L", "D": "D", "A": "W"})
    
    all_matches = pd.concat([
        home_matches[["date", "team_goals", "opp_goals", "team_result"]],
        away_matches[["date", "team_goals", "opp_goals", "team_result"]]
    ]).sort_values("date", ascending=False).head(n_matches)
    
    if all_matches.empty:
        return {
            "form_wins": 0, "form_draws": 0, "form_losses": 0,
            "form_gf": 0, "form_ga": 0, "form_pts": 0,
            "form_str": "", "form_matches": 0
        }
    
    wins = (all_matches["team_result"] == "W").sum()
    draws = (all_matches["team_result"] == "D").sum()
    losses = (all_matches["team_result"] == "L").sum()
    gf = all_matches["team_goals"].sum()
    ga = all_matches["opp_goals"].sum()
    pts = wins * 3 + draws
    form_str = "".join(all_matches["team_result"].values)
    
    return {
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
        "form_gf": gf,
        "form_ga": ga,
        "form_pts": pts,
        "form_str": form_str,
        "form_matches": len(all_matches)
    }


def calculate_head2head(df: pd.DataFrame, team1: str, team2: str,
                         date: pd.Timestamp, n_matches: int = 5) -> dict:
    """Calcula estadísticas head-to-head entre dos equipos."""
    h2h = df[
        ((df["home_team"] == team1) & (df["away_team"] == team2) |
         (df["home_team"] == team2) & (df["away_team"] == team1)) &
        (df["date"] < date)
    ].sort_values("date", ascending=False).head(n_matches)
    
    if h2h.empty:
        return {"h2h_team1_wins": 0, "h2h_team2_wins": 0, "h2h_draws": 0, "h2h_matches": 0}
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    
    for _, match in h2h.iterrows():
        if match["result"] == "D":
            draws += 1
        elif match["home_team"] == team1 and match["result"] == "H":
            team1_wins += 1
        elif match["away_team"] == team1 and match["result"] == "A":
            team1_wins += 1
        else:
            team2_wins += 1
    
    return {
        "h2h_team1_wins": team1_wins,
        "h2h_team2_wins": team2_wins,
        "h2h_draws": draws,
        "h2h_matches": len(h2h)
    }


def calculate_season_stats(df: pd.DataFrame, team: str, date: pd.Timestamp,
                            season: int) -> dict:
    """Calcula stats acumuladas del equipo en la temporada hasta la fecha dada."""
    season_data = df[(df["season"] == season) & (df["date"] < date)]
    
    # Como local
    home = season_data[season_data["home_team"] == team]
    home_gf = home["home_goals"].sum()
    home_ga = home["away_goals"].sum()
    home_wins = (home["result"] == "H").sum()
    home_played = len(home)
    
    # Como visitante
    away = season_data[season_data["away_team"] == team]
    away_gf = away["away_goals"].sum()
    away_ga = away["home_goals"].sum()
    away_wins = (away["result"] == "A").sum()
    away_played = len(away)
    
    total_played = home_played + away_played
    
    return {
        "season_gf": home_gf + away_gf,
        "season_ga": home_ga + away_ga,
        "season_gd": (home_gf + away_gf) - (home_ga + away_ga),
        "season_wins": home_wins + away_wins,
        "season_played": total_played,
        "season_win_rate": (home_wins + away_wins) / max(total_played, 1),
        "season_avg_gf": (home_gf + away_gf) / max(total_played, 1),
        "season_avg_ga": (home_ga + away_ga) / max(total_played, 1),
        "home_win_rate": home_wins / max(home_played, 1),
        "away_win_rate": away_wins / max(away_played, 1),
    }


def build_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """
    Construye el DataFrame de features para entrenamiento del modelo.
    
    Para cada partido, calcula features del equipo local y visitante:
    - Forma reciente
    - Stats acumuladas de la temporada
    - Head-to-head
    - ELO rating (si disponible)
    
    Returns:
        DataFrame con features + target (result)
    """
    print("Construyendo features...")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    features_list = []
    total = len(df)
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"  Procesando {idx}/{total}...")
        
        date = row["date"]
        season = row["season"]
        home = row["home_team"]
        away = row["away_team"]
        
        # Forma reciente
        home_form = calculate_form(df, home, date, form_window)
        away_form = calculate_form(df, away, date, form_window)
        
        # Stats temporada
        home_season = calculate_season_stats(df, home, date, season)
        away_season = calculate_season_stats(df, away, date, season)
        
        # Head-to-head
        h2h = calculate_head2head(df, home, away, date)
        
        # Construir fila de features
        feature_row = {
            "date": date,
            "season": season,
            "home_team": home,
            "away_team": away,
            # Target
            "result": row["result"],
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            # Home form
            "home_form_pts": home_form["form_pts"],
            "home_form_gf": home_form["form_gf"],
            "home_form_ga": home_form["form_ga"],
            "home_form_wins": home_form["form_wins"],
            # Away form
            "away_form_pts": away_form["form_pts"],
            "away_form_gf": away_form["form_gf"],
            "away_form_ga": away_form["form_ga"],
            "away_form_wins": away_form["form_wins"],
            # Home season stats
            "home_season_win_rate": home_season["season_win_rate"],
            "home_season_avg_gf": home_season["season_avg_gf"],
            "home_season_avg_ga": home_season["season_avg_ga"],
            "home_season_gd": home_season["season_gd"],
            "home_home_win_rate": home_season["home_win_rate"],
            # Away season stats
            "away_season_win_rate": away_season["season_win_rate"],
            "away_season_avg_gf": away_season["season_avg_gf"],
            "away_season_avg_ga": away_season["season_avg_ga"],
            "away_season_gd": away_season["season_gd"],
            "away_away_win_rate": away_season["away_win_rate"],
            # H2H
            "h2h_home_wins": h2h["h2h_team1_wins"],
            "h2h_away_wins": h2h["h2h_team2_wins"],
            "h2h_draws": h2h["h2h_draws"],
        }
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    
    # Guardar
    output_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(output_path, index=False)
    print(f"Features guardadas: {output_path} ({len(features_df)} filas, {len(features_df.columns)} columnas)")
    
    return features_df


if __name__ == "__main__":
    print("=" * 60)
    print("  Feature Engineering — Liga BetPlay")
    print("=" * 60)
    
    # Cargar datos limpios
    data_path = PROCESSED_DIR / "matches_clean.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        csv_path = PROCESSED_DIR / "matches_clean.csv"
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Usar solo últimas 3 temporadas para rapidez
    recent = df[df["season"] >= 2023]
    features = build_features(recent)
    print(f"\nFeatures generadas: {features.shape}")
    print(features.head())
