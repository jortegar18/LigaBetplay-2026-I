"""
data_collector.py — Liga BetPlay Data Collection
Descarga datos de la liga colombiana desde FBref via web scraping.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def get_fbref_season_url(season_end_year: int) -> str:
    """Genera la URL de FBref para una temporada específica de la Primera A."""
    return f"https://fbref.com/en/comps/41/{season_end_year}/schedule/{season_end_year}-Primera-A-Scores-and-Fixtures"


def scrape_fbref_standings(season_end_year: int) -> pd.DataFrame:
    """
    Scrape tabla de posiciones de FBref para una temporada de Primera A.
    
    Args:
        season_end_year: Año final de la temporada (ej: 2024)
    
    Returns:
        DataFrame con tabla de posiciones
    """
    url = f"https://fbref.com/en/comps/41/{season_end_year}/{season_end_year}-Primera-A-Stats"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        if tables:
            df = tables[0]
            df["season"] = season_end_year
            return df
        else:
            print(f"No se encontraron tablas para {season_end_year}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error scraping temporada {season_end_year}: {e}")
        return pd.DataFrame()


def scrape_fbref_fixtures(season_end_year: int) -> pd.DataFrame:
    """
    Scrape resultados/fixtures de FBref para una temporada.
    
    Args:
        season_end_year: Año final de la temporada
    
    Returns:
        DataFrame con fixtures y resultados
    """
    url = get_fbref_season_url(season_end_year)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        tables = pd.read_html(response.text)
        if tables:
            df = tables[0]
            df["season"] = season_end_year
            return df
        else:
            print(f"No se encontraron fixtures para {season_end_year}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error scraping fixtures {season_end_year}: {e}")
        return pd.DataFrame()


def collect_multiple_seasons(start_year: int = 2019, end_year: int = 2025,
                              data_type: str = "standings") -> pd.DataFrame:
    """
    Recopila datos de múltiples temporadas de FBref.
    
    Args:
        start_year: Año de inicio
        end_year: Año final
        data_type: 'standings' o 'fixtures'
    
    Returns:
        DataFrame combinado de todas las temporadas
    """
    all_data = []
    scrape_func = scrape_fbref_standings if data_type == "standings" else scrape_fbref_fixtures
    
    for year in range(start_year, end_year + 1):
        print(f"Descargando {data_type} temporada {year}...")
        df = scrape_func(year)
        if not df.empty:
            all_data.append(df)
        # Respetar rate limits de FBref (3 seg entre requests)
        time.sleep(4)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Guardar copia cruda
        output_path = RAW_DIR / f"fbref_{data_type}_combined.csv"
        combined.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Guardado en {output_path} ({len(combined)} filas)")
        return combined
    
    return pd.DataFrame()


def load_kaggle_data(filepath: str) -> pd.DataFrame:
    """
    Carga un dataset de Kaggle descargado manualmente.
    
    Args:
        filepath: Ruta al archivo CSV
    
    Returns:
        DataFrame con los datos
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Archivo no encontrado: {filepath}")
        print("Descarga el dataset de Kaggle manualmente:")
        print("  - 'Primera A - Colômbia (2007 a 2022)'")
        print("  - 'Colombian Soccer Database 2026-1'")
        print(f"Y colócalo en: {RAW_DIR}")
        return pd.DataFrame()
    
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    return df


def generate_sample_data() -> pd.DataFrame:
    """
    Genera datos de ejemplo de la Liga BetPlay para desarrollo y testing.
    Incluye resultados realistas con equipos colombianos reales.
    """
    import numpy as np
    
    teams = [
        "Atlético Nacional", "Millonarios", "América de Cali",
        "Deportivo Cali", "Junior", "Santa Fe", "Medellín",
        "Once Caldas", "Deportivo Pereira", "Tolima",
        "Bucaramanga", "Alianza Petrolera", "Envigado",
        "La Equidad", "Jaguares", "Patriotas",
        "Águilas Doradas", "Boyacá Chicó", "Pasto",
        "Unión Magdalena"
    ]
    
    np.random.seed(42)
    records = []
    
    for season in range(2015, 2026):
        for half in ["Apertura", "Finalización"]:
            # Cada equipo juega contra todos una vez en cada fase
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i == j:
                        continue
                    
                    # Simular resultado con sesgo local
                    home_goals = np.random.poisson(1.4)
                    away_goals = np.random.poisson(1.0)
                    
                    if home_goals == away_goals:
                        result = "D"
                    elif home_goals > away_goals:
                        result = "H"
                    else:
                        result = "A"
                    
                    date_offset = np.random.randint(0, 180)
                    base_month = 1 if half == "Apertura" else 7
                    
                    records.append({
                        "season": season,
                        "phase": half,
                        "date": f"{season}-{base_month + date_offset // 30:02d}-{(date_offset % 28) + 1:02d}",
                        "home_team": home,
                        "away_team": away,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "result": result,
                    })
    
    df = pd.DataFrame(records)
    
    # Guardar datos de ejemplo
    output_path = RAW_DIR / "liga_betplay_sample.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Datos de ejemplo generados: {len(df)} partidos → {output_path}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Collector — Liga BetPlay Colombia")
    print("=" * 60)
    
    # Generar datos de ejemplo siempre
    print("\n📦 Generando datos de ejemplo...")
    sample_df = generate_sample_data()
    print(f"   {len(sample_df)} partidos generados\n")
    
    # Intentar scraping de FBref (puede fallar por rate limits)
    print("🌐 Intentando scraping de FBref...")
    try:
        standings = collect_multiple_seasons(2022, 2024, "standings")
        if not standings.empty:
            print(f"   Standings: {len(standings)} filas")
    except Exception as e:
        print(f"   Scraping falló (normal si hay rate limits): {e}")
        print("   Los datos de ejemplo están disponibles para continuar.")
