"""
data_cleaner.py — Limpieza y transformación de datos de Liga BetPlay
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Mapeo de nombres inconsistentes de equipos
TEAM_NAME_MAP = {
    "Atletico Nacional": "Atlético Nacional",
    "Nacional": "Atlético Nacional",
    "Atl. Nacional": "Atlético Nacional",
    "America de Cali": "América de Cali",
    "America": "América de Cali",
    "Dep. Cali": "Deportivo Cali",
    "Cali": "Deportivo Cali",
    "Ind. Medellin": "Medellín",
    "Independiente Medellin": "Medellín",
    "Independiente Medellín": "Medellín",
    "DIM": "Medellín",
    "Ind. Santa Fe": "Santa Fe",
    "Independiente Santa Fe": "Santa Fe",
    "Dep. Pereira": "Deportivo Pereira",
    "Deportes Tolima": "Tolima",
    "Atletico Bucaramanga": "Bucaramanga",
    "Atl. Bucaramanga": "Bucaramanga",
    "Boyaca Chico": "Boyacá Chicó",
    "Deportivo Pasto": "Pasto",
    "AD Pasto": "Pasto",
    "Aguilas Doradas": "Águilas Doradas",
    "Alianza FC": "Alianza Petrolera",
    "Cortulua": "Cortuluá",
    "Union Magdalena": "Unión Magdalena",
}


def normalize_team_name(name: str) -> str:
    """Normaliza el nombre de un equipo colombiano."""
    if pd.isna(name):
        return name
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def clean_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza un DataFrame de partidos.
    
    Espera columnas: home_team, away_team, home_goals, away_goals, date, season
    """
    df = df.copy()
    
    # Normalizar nombres
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(normalize_team_name)
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(normalize_team_name)
    
    # Asegurar tipos numéricos
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Crear fecha como datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Calcular resultado si no existe
    if "result" not in df.columns and "home_goals" in df.columns:
        df["result"] = np.where(
            df["home_goals"] > df["away_goals"], "H",
            np.where(df["home_goals"] == df["away_goals"], "D", "A")
        )
    
    # Calcular gol diferencia
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
        df["total_goals"] = df["home_goals"] + df["away_goals"]
    
    # Eliminar filas con datos críticos faltantes
    critical_cols = ["home_team", "away_team", "home_goals", "away_goals"]
    existing_critical = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=existing_critical)
    
    # Ordenar por fecha
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    
    return df


def add_points(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de puntos por partido (home_pts, away_pts)."""
    df = df.copy()
    
    df["home_pts"] = df["result"].map({"H": 3, "D": 1, "A": 0})
    df["away_pts"] = df["result"].map({"H": 0, "D": 1, "A": 3})
    
    return df


def build_standings(df: pd.DataFrame, season: int = None, 
                     phase: str = None) -> pd.DataFrame:
    """
    Construye tabla de posiciones a partir de resultados de partidos.
    
    Args:
        df: DataFrame con partidos
        season: Filtrar por temporada
        phase: Filtrar por fase (Apertura/Finalización)
    
    Returns:
        DataFrame con tabla de posiciones ordenada
    """
    data = df.copy()
    
    if season is not None:
        data = data[data["season"] == season]
    if phase is not None and "phase" in data.columns:
        data = data[data["phase"] == phase]
    
    if "home_pts" not in data.columns:
        data = add_points(data)
    
    # Stats como local
    home_stats = data.groupby("home_team").agg(
        home_played=("result", "count"),
        home_wins=("result", lambda x: (x == "H").sum()),
        home_draws=("result", lambda x: (x == "D").sum()),
        home_losses=("result", lambda x: (x == "A").sum()),
        home_gf=("home_goals", "sum"),
        home_ga=("away_goals", "sum"),
        home_pts=("home_pts", "sum"),
    ).reset_index().rename(columns={"home_team": "team"})
    
    # Stats como visitante
    away_stats = data.groupby("away_team").agg(
        away_played=("result", "count"),
        away_wins=("result", lambda x: (x == "A").sum()),
        away_draws=("result", lambda x: (x == "D").sum()),
        away_losses=("result", lambda x: (x == "H").sum()),
        away_gf=("away_goals", "sum"),
        away_ga=("home_goals", "sum"),
        away_pts=("away_pts", "sum"),
    ).reset_index().rename(columns={"away_team": "team"})
    
    # Merge
    standings = pd.merge(home_stats, away_stats, on="team", how="outer").fillna(0)
    
    # Totales
    standings["played"] = standings["home_played"] + standings["away_played"]
    standings["wins"] = standings["home_wins"] + standings["away_wins"]
    standings["draws"] = standings["home_draws"] + standings["away_draws"]
    standings["losses"] = standings["home_losses"] + standings["away_losses"]
    standings["gf"] = standings["home_gf"] + standings["away_gf"]
    standings["ga"] = standings["home_ga"] + standings["away_ga"]
    standings["gd"] = standings["gf"] - standings["ga"]
    standings["pts"] = standings["home_pts"] + standings["away_pts"]
    
    # Ordenar
    standings = standings.sort_values(
        ["pts", "gd", "gf"], ascending=[False, False, False]
    ).reset_index(drop=True)
    standings.index = standings.index + 1  # Posición desde 1
    standings.index.name = "pos"
    
    return standings[["team", "played", "wins", "draws", "losses", 
                       "gf", "ga", "gd", "pts"]]


def process_and_save(input_path: str = None) -> pd.DataFrame:
    """
    Pipeline completo: cargar, limpiar, y guardar datos procesados.
    
    Args:
        input_path: Ruta al CSV crudo. Si None, usa el archivo de ejemplo.
    
    Returns:
        DataFrame limpio
    """
    if input_path is None:
        input_path = RAW_DIR / "liga_betplay_sample.csv"
    
    print(f"Cargando datos de {input_path}...")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    
    print("Limpiando datos...")
    df_clean = clean_match_data(df)
    df_clean = add_points(df_clean)
    
    # Guardar como Parquet
    output_path = PROCESSED_DIR / "matches_clean.parquet"
    df_clean.to_parquet(output_path, index=False)
    print(f"Datos limpios guardados: {output_path} ({len(df_clean)} filas)")
    
    # Guardar también como CSV para inspección fácil
    csv_path = PROCESSED_DIR / "matches_clean.csv"
    df_clean.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    return df_clean


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Cleaner — Liga BetPlay Colombia")
    print("=" * 60)
    
    df = process_and_save()
    
    print(f"\nResumen:")
    print(f"  Partidos: {len(df)}")
    print(f"  Temporadas: {df['season'].nunique()}")
    print(f"  Equipos: {df['home_team'].nunique()}")
    
    # Ejemplo: tabla de posiciones 2024
    print("\n📊 Tabla de posiciones 2024 (ejemplo):")
    standings = build_standings(df, season=2024)
    print(standings.head(10).to_string())
