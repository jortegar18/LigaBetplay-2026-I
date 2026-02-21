"""
elo_rating.py — Sistema de Rating ELO para Liga BetPlay
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


class EloSystem:
    """
    Sistema de rating ELO adaptado para fútbol colombiano.
    
    Características:
    - Factor K adaptativo (más alto al inicio, se estabiliza)
    - Bonificación por diferencia de goles
    - Ventaja de local configurable
    """
    
    def __init__(self, initial_elo: float = 1500, k_factor: float = 32,
                 home_advantage: float = 65):
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
        self.history = []
    
    def get_rating(self, team: str) -> float:
        """Obtiene el rating actual de un equipo."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float, 
                        home_advantage: float = None) -> float:
        """Calcula el score esperado para el equipo A."""
        if home_advantage is None:
            home_advantage = self.home_advantage
        adjusted_a = rating_a + home_advantage
        return 1 / (1 + 10 ** ((rating_b - adjusted_a) / 400))
    
    def goal_factor(self, goal_diff: int) -> float:
        """Factor multiplicador basado en diferencia de goles."""
        abs_diff = abs(goal_diff)
        if abs_diff <= 1:
            return 1.0
        elif abs_diff == 2:
            return 1.5
        else:
            return (11 + abs_diff) / 8
    
    def update(self, home_team: str, away_team: str, 
               home_goals: int, away_goals: int, 
               date=None, season=None) -> dict:
        """
        Actualiza ratings después de un partido.
        
        Returns:
            Dict con ratings antes y después del partido
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Score real (1=victoria, 0.5=empate, 0=derrota)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0
        
        actual_away = 1.0 - actual_home
        
        # Scores esperados
        expected_home = self.expected_score(home_elo, away_elo)
        expected_away = 1.0 - expected_home
        
        # Factor de goles
        gf = self.goal_factor(home_goals - away_goals)
        
        # Actualizar ratings
        k = self.k_factor
        home_new = home_elo + k * gf * (actual_home - expected_home)
        away_new = away_elo + k * gf * (actual_away - expected_away)
        
        self.ratings[home_team] = home_new
        self.ratings[away_team] = away_new
        
        record = {
            "date": date,
            "season": season,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_elo_before": home_elo,
            "away_elo_before": away_elo,
            "home_elo_after": home_new,
            "away_elo_after": away_new,
            "home_expected": expected_home,
            "home_change": home_new - home_elo,
        }
        self.history.append(record)
        
        return record
    
    def process_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa un DataFrame completo de partidos y calcula ELO para todos.
        
        Args:
            df: DataFrame ordenado por fecha con: home_team, away_team, 
                home_goals, away_goals, date, season
        
        Returns:
            DataFrame con historial de ELO
        """
        df = df.sort_values("date").reset_index(drop=True)
        
        for _, row in df.iterrows():
            self.update(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                date=row.get("date"),
                season=row.get("season"),
            )
        
        return pd.DataFrame(self.history)
    
    def get_rankings(self) -> pd.DataFrame:
        """Devuelve ranking actual de todos los equipos."""
        rankings = pd.DataFrame([
            {"team": team, "elo": rating}
            for team, rating in self.ratings.items()
        ]).sort_values("elo", ascending=False).reset_index(drop=True)
        rankings.index = rankings.index + 1
        rankings.index.name = "rank"
        return rankings
    
    def predict_match(self, home_team: str, away_team: str) -> dict:
        """
        Predice el resultado de un partido basado en ELO.
        
        Returns:
            Dict con probabilidades de victoria/empate/derrota
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        expected_home = self.expected_score(home_elo, away_elo)
        
        # Aproximación: convertir expected score a probabilidades W/D/L
        # Ajuste empírico para fútbol
        draw_prob = max(0.15, 0.38 - abs(expected_home - 0.5) * 0.8)
        remaining = 1 - draw_prob
        home_win_prob = expected_home * remaining / (expected_home + (1 - expected_home))
        away_win_prob = remaining - home_win_prob
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "home_win": round(home_win_prob, 3),
            "draw": round(draw_prob, 3),
            "away_win": round(away_win_prob, 3),
        }


def build_elo_system(df: pd.DataFrame) -> EloSystem:
    """
    Construye y entrena el sistema ELO con datos históricos.
    
    Args:
        df: DataFrame con partidos limpios
    
    Returns:
        EloSystem entrenado
    """
    elo = EloSystem(initial_elo=1500, k_factor=32, home_advantage=65)
    
    print("Calculando ratings ELO...")
    history_df = elo.process_matches(df)
    
    # Guardar historial
    output_path = PROCESSED_DIR / "elo_history.parquet"
    history_df.to_parquet(output_path, index=False)
    print(f"Historial ELO: {output_path}")
    
    # Guardar rankings actuales
    rankings = elo.get_rankings()
    rankings_path = PROCESSED_DIR / "elo_rankings.csv"
    rankings.to_csv(rankings_path, encoding="utf-8-sig")
    print(f"Rankings: {rankings_path}")
    
    return elo


if __name__ == "__main__":
    print("=" * 60)
    print("  ELO Rating System — Liga BetPlay")
    print("=" * 60)
    
    data_path = PROCESSED_DIR / "matches_clean.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(PROCESSED_DIR / "matches_clean.csv", encoding="utf-8-sig")
    
    elo = build_elo_system(df)
    
    print("\n🏆 Top 10 Rankings ELO:")
    print(elo.get_rankings().head(10).to_string())
    
    print("\n⚽ Predicción ejemplo:")
    pred = elo.predict_match("Atlético Nacional", "Millonarios")
    print(f"  {pred['home_team']} ({pred['home_elo']}) vs "
          f"{pred['away_team']} ({pred['away_elo']})")
    print(f"  Local: {pred['home_win']:.1%} | Empate: {pred['draw']:.1%} | "
          f"Visitante: {pred['away_win']:.1%}")
