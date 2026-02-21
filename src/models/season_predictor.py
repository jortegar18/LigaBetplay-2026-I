"""
season_predictor.py — Predicción de tabla final Liga BetPlay I-2026
Simula los partidos restantes usando ELO ratings y Monte Carlo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.real_data_2026 import (
    get_standings_df, get_results_df, TEAMS_2026, STANDINGS_2026
)
from src.models.elo_rating import EloSystem

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TOTAL_FECHAS = 19  # Total de fechas en la fase regular
CURRENT_FECHA = 7


def generate_remaining_fixtures(teams: list, current_fecha: int,
                                  played_results: pd.DataFrame) -> list:
    """
    Genera los fixtures restantes de la temporada.
    En la Liga BetPlay Apertura, cada equipo juega contra todos una vez
    (todos contra todos, ida). 20 equipos = 19 fechas, 10 partidos por fecha.
    """
    # Pares ya jugados (home, away)
    played = set()
    for _, row in played_results.iterrows():
        played.add((row["home_team"], row["away_team"]))
    
    # Generar todos los pares posibles
    remaining = []
    for home in teams:
        for away in teams:
            if home != away and (home, away) not in played and (away, home) not in played:
                remaining.append({"home_team": home, "away_team": away})
    
    # Distribuir en fechas
    np.random.shuffle(remaining)
    fixtures_by_fecha = []
    fecha = current_fecha + 1
    
    for i in range(0, len(remaining), 10):
        batch = remaining[i:i+10]
        for match in batch:
            match["fecha"] = fecha
        fixtures_by_fecha.extend(batch)
        fecha += 1
    
    return fixtures_by_fecha


def simulate_match_elo(elo_system: EloSystem, home_team: str, 
                        away_team: str) -> tuple:
    """
    Simula un partido usando el sistema ELO.
    
    Returns:
        (home_goals, away_goals)
    """
    prediction = elo_system.predict_match(home_team, away_team)
    
    # Generar resultado basado en probabilidades
    rand = np.random.random()
    
    if rand < prediction["home_win"]:
        # Victoria local
        home_goals = np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25])
        away_goals = np.random.choice([0, 1], p=[0.6, 0.4])
        if away_goals >= home_goals:
            away_goals = home_goals - 1
    elif rand < prediction["home_win"] + prediction["draw"]:
        # Empate
        goals = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
        home_goals = away_goals = goals
    else:
        # Victoria visitante
        away_goals = np.random.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
        home_goals = np.random.choice([0, 1], p=[0.55, 0.45])
        if home_goals >= away_goals:
            home_goals = away_goals - 1
    
    return max(0, int(home_goals)), max(0, int(away_goals))


def simulate_season(results_df: pd.DataFrame, remaining_fixtures: list,
                      n_simulations: int = 1000) -> pd.DataFrame:
    """
    Simula el resto de la temporada N veces con Monte Carlo.
    
    Returns:
        DataFrame con probabilidades por equipo: avg_pts, avg_pos,
        prob_top8, prob_champion, etc.
    """
    print(f"🎲 Simulando {n_simulations} temporadas...")
    
    # Calcular ELO con resultados actuales
    base_elo = EloSystem(initial_elo=1500, k_factor=40, home_advantage=70)
    base_elo.process_matches(results_df)
    
    # Puntos actuales
    current_pts = {}
    current_gf = {}
    current_ga = {}
    for row in STANDINGS_2026:
        team = row[1]
        current_pts[team] = row[9]  # pts
        current_gf[team] = row[6]   # gf
        current_ga[team] = row[7]   # gc
    
    all_results = {team: {"pts": [], "pos": [], "gf": [], "ga": []} 
                   for team in TEAMS_2026}
    
    # Matriz de distribución de posiciones: equipo -> posición -> conteo
    n_teams = len(TEAMS_2026)
    pos_counts = {team: np.zeros(n_teams, dtype=int) for team in TEAMS_2026}
    
    for sim in range(n_simulations):
        if sim % 200 == 0:
            print(f"  Simulación {sim}/{n_simulations}...")
        
        # Copiar puntos actuales
        pts = dict(current_pts)
        gf = dict(current_gf)
        ga = dict(current_ga)
        
        # Simular partidos restantes
        for fixture in remaining_fixtures:
            home = fixture["home_team"]
            away = fixture["away_team"]
            
            hg, ag = simulate_match_elo(base_elo, home, away)
            
            gf[home] = gf.get(home, 0) + hg
            ga[home] = ga.get(home, 0) + ag
            gf[away] = gf.get(away, 0) + ag
            ga[away] = ga.get(away, 0) + hg
            
            if hg > ag:
                pts[home] += 3
            elif hg == ag:
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3
        
        # Calcular posiciones
        standings = sorted(TEAMS_2026, 
                          key=lambda t: (pts.get(t, 0), gf.get(t, 0) - ga.get(t, 0), gf.get(t, 0)),
                          reverse=True)
        
        for pos, team in enumerate(standings, 1):
            all_results[team]["pts"].append(pts.get(team, 0))
            all_results[team]["pos"].append(pos)
            all_results[team]["gf"].append(gf.get(team, 0))
            all_results[team]["ga"].append(ga.get(team, 0))
            pos_counts[team][pos - 1] += 1
    
    # Compilar resultados
    summary = []
    for team in TEAMS_2026:
        pts_arr = np.array(all_results[team]["pts"])
        pos_arr = np.array(all_results[team]["pos"])
        
        summary.append({
            "team": team,
            "current_pts": current_pts.get(team, 0),
            "avg_final_pts": pts_arr.mean(),
            "min_pts": pts_arr.min(),
            "max_pts": pts_arr.max(),
            "avg_pos": pos_arr.mean(),
            "best_pos": pos_arr.min(),
            "worst_pos": pos_arr.max(),
            "prob_top1": (pos_arr == 1).mean(),
            "prob_top4": (pos_arr <= 4).mean(),
            "prob_top8": (pos_arr <= 8).mean(),
            "prob_bottom4": (pos_arr >= 17).mean(),
        })
    
    summary_df = pd.DataFrame(summary).sort_values("avg_pos")
    summary_df.index = range(1, len(summary_df) + 1)
    summary_df.index.name = "predicted_pos"
    
    # Crear matriz de probabilidades de posición (equipo × posición)
    pos_matrix = pd.DataFrame(
        {team: pos_counts[team] / n_simulations for team in TEAMS_2026},
        index=range(1, n_teams + 1)
    ).T
    pos_matrix.columns = [f"pos_{i}" for i in range(1, n_teams + 1)]
    pos_matrix.index.name = "team"
    
    # Ordenar por posición promedio
    team_order = summary_df["team"].tolist()
    pos_matrix = pos_matrix.loc[team_order]
    
    # Guardar
    output_path = PROCESSED_DIR / "season_predictions.csv"
    summary_df.to_csv(output_path, encoding="utf-8-sig")
    
    heatmap_path = PROCESSED_DIR / "position_heatmap.csv"
    pos_matrix.to_csv(heatmap_path, encoding="utf-8-sig")
    
    print(f"✅ Predicciones guardadas: {output_path}")
    print(f"✅ Mapa de calor guardado: {heatmap_path}")
    
    return summary_df


def predict_season(n_simulations: int = 1000) -> pd.DataFrame:
    """Pipeline completo de predicción de temporada."""
    results_df = get_results_df()
    remaining = generate_remaining_fixtures(TEAMS_2026, CURRENT_FECHA, results_df)
    
    print(f"📊 Partidos jugados: {len(results_df)}")
    print(f"📊 Partidos por jugar: {len(remaining)}")
    print(f"📊 Fechas restantes: {TOTAL_FECHAS - CURRENT_FECHA}")
    
    predictions = simulate_season(results_df, remaining, n_simulations)
    return predictions


if __name__ == "__main__":
    print("=" * 60)
    print("  Season Predictor — Liga BetPlay I-2026")
    print("=" * 60)
    
    predictions = predict_season(n_simulations=500)
    
    print("\n🔮 Predicción de tabla final:")
    print(predictions[["team", "current_pts", "avg_final_pts", "avg_pos", 
                        "prob_top8", "prob_top1"]].round(3).to_string())
