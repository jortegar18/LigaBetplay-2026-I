"""
live_scraper.py — Scraper gratuito para datos en vivo Liga BetPlay I-2026
Fuentes: FBref (free, no API key needed)

Descarga la tabla de posiciones y resultados automáticamente.
"""

import requests
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# FBref URLs para Liga BetPlay
FBREF_LEAGUE_URL = "https://fbref.com/en/comps/41/Categoria-Primera-A-Stats"
FBREF_SCHEDULE_URL = "https://fbref.com/en/comps/41/schedule/Categoria-Primera-A-Scores-and-Fixtures"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CACHE_HOURS = 4


def _get_cached(cache_name: str, max_age_hours: int = CACHE_HOURS):
    """Lee datos de caché si existen y no han expirado."""
    cache_file = CACHE_DIR / f"{cache_name}.json"
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=max_age_hours):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _save_cache(cache_name: str, data):
    """Guarda datos en caché."""
    cache_file = CACHE_DIR / f"{cache_name}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scrape_standings() -> pd.DataFrame:
    """
    Scrape la tabla de posiciones actual desde FBref.
    Returns: DataFrame con pos, team, pj, g, e, p, gf, gc, dif, pts
    """
    cache_key = "live_standings"
    cached = _get_cached(cache_key)
    
    if cached is not None:
        print("📋 Usando standings de caché")
        return pd.DataFrame(cached)
    
    print("📡 Descargando tabla de posiciones de FBref...")
    
    try:
        resp = requests.get(FBREF_LEAGUE_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Error descargando standings: {e}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Buscar tabla de standings (overall)
    tables = pd.read_html(resp.text)
    
    standings_df = None
    for table in tables:
        cols = [str(c).lower() for c in table.columns.get_level_values(-1)]
        if "pts" in cols and ("w" in cols or "g" in cols):
            standings_df = table
            break
    
    if standings_df is None:
        print("⚠️ No se encontró tabla de posiciones")
        return pd.DataFrame()
    
    # Limpiar columnas (pueden ser multi-level)
    if isinstance(standings_df.columns, pd.MultiIndex):
        standings_df.columns = standings_df.columns.get_level_values(-1)
    
    # Normalizar nombres de columnas
    col_map = {}
    for c in standings_df.columns:
        cl = str(c).lower().strip()
        if cl == "rk": col_map[c] = "pos"
        elif cl == "squad": col_map[c] = "team"
        elif cl == "mp": col_map[c] = "pj"
        elif cl == "w": col_map[c] = "g"
        elif cl == "d": col_map[c] = "e"
        elif cl == "l": col_map[c] = "p"
        elif cl == "gf": col_map[c] = "gf"
        elif cl == "ga": col_map[c] = "gc"
        elif cl == "gd": col_map[c] = "dif"
        elif cl == "pts": col_map[c] = "pts"
    
    standings_df = standings_df.rename(columns=col_map)
    
    # Seleccionar solo columnas relevantes
    keep_cols = ["pos", "team", "pj", "g", "e", "p", "gf", "gc", "dif", "pts"]
    available = [c for c in keep_cols if c in standings_df.columns]
    standings_df = standings_df[available].copy()
    
    # Limpiar tipos
    for c in ["pos", "pj", "g", "e", "p", "gf", "gc", "dif", "pts"]:
        if c in standings_df.columns:
            standings_df[c] = pd.to_numeric(standings_df[c], errors="coerce")
    
    # Guardar caché
    _save_cache(cache_key, standings_df.to_dict(orient="records"))
    
    # Guardar en processed
    standings_df.to_csv(PROCESSED_DIR / "standings_live.csv", 
                        index=False, encoding="utf-8-sig")
    
    print(f"✅ Tabla cargada: {len(standings_df)} equipos")
    return standings_df


def scrape_results() -> pd.DataFrame:
    """
    Scrape todos los resultados de la temporada desde FBref.
    Returns: DataFrame con fecha, date, home_team, away_team, home_goals, away_goals, etc.
    """
    cache_key = "live_results"
    cached = _get_cached(cache_key, max_age_hours=2)
    
    if cached is not None:
        print("📋 Usando resultados de caché")
        df = pd.DataFrame(cached)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    
    print("📡 Descargando resultados de FBref...")
    
    try:
        resp = requests.get(FBREF_SCHEDULE_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Error descargando resultados: {e}")
        return pd.DataFrame()
    
    tables = pd.read_html(resp.text)
    schedule_df = None
    
    for table in tables:
        cols = [str(c).lower() for c in table.columns.get_level_values(-1)]
        if "home" in cols and "away" in cols:
            schedule_df = table
            break
    
    if schedule_df is None:
        print("⚠️ No se encontró tabla de fixtures")
        return pd.DataFrame()
    
    if isinstance(schedule_df.columns, pd.MultiIndex):
        schedule_df.columns = schedule_df.columns.get_level_values(-1)
    
    # Renombrar columnas
    col_map = {}
    for c in schedule_df.columns:
        cl = str(c).lower().strip()
        if cl == "wk": col_map[c] = "fecha"
        elif cl == "day": col_map[c] = "day"
        elif cl == "date": col_map[c] = "date"
        elif cl == "home": col_map[c] = "home_team"
        elif cl == "away": col_map[c] = "away_team"
        elif cl == "score": col_map[c] = "score"
    
    schedule_df = schedule_df.rename(columns=col_map)
    
    # Filtrar solo partidos jugados (que tienen score)
    if "score" in schedule_df.columns:
        played = schedule_df[schedule_df["score"].notna() & 
                             schedule_df["score"].astype(str).str.contains(r"\d+", regex=True)].copy()
    else:
        print("⚠️ Columna 'score' no encontrada")
        return pd.DataFrame()
    
    if played.empty:
        print("⚠️ No se encontraron partidos jugados")
        return pd.DataFrame()
    
    # Parsear score "2–1" -> home_goals, away_goals
    score_split = played["score"].astype(str).str.extract(r"(\d+)\D+(\d+)")
    played["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
    played["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")
    
    # Limpiar
    played = played.dropna(subset=["home_goals", "away_goals"])
    played["home_goals"] = played["home_goals"].astype(int)
    played["away_goals"] = played["away_goals"].astype(int)
    
    if "date" in played.columns:
        played["date"] = pd.to_datetime(played["date"], errors="coerce")
    
    if "fecha" in played.columns:
        played["fecha"] = pd.to_numeric(played["fecha"], errors="coerce").fillna(0).astype(int)
    
    # Calcular resultado
    played["result"] = np.where(
        played["home_goals"] > played["away_goals"], "H",
        np.where(played["home_goals"] == played["away_goals"], "D", "A")
    )
    played["total_goals"] = played["home_goals"] + played["away_goals"]
    played["season"] = 2026
    played["phase"] = "Apertura"
    
    # Seleccionar columnas
    keep = ["fecha", "date", "home_team", "away_team", "home_goals", 
            "away_goals", "result", "total_goals", "season", "phase"]
    available = [c for c in keep if c in played.columns]
    result_df = played[available].reset_index(drop=True)
    
    # Guardar caché
    cache_data = result_df.copy()
    if "date" in cache_data.columns:
        cache_data["date"] = cache_data["date"].astype(str)
    _save_cache(cache_key, cache_data.to_dict(orient="records"))
    
    # Guardar en processed
    result_df.to_csv(RAW_DIR / "liga_betplay_2026_live.csv", 
                     index=False, encoding="utf-8-sig")
    
    print(f"✅ Resultados cargados: {len(result_df)} partidos")
    return result_df


def scrape_upcoming() -> pd.DataFrame:
    """
    Obtiene los próximos partidos desde FBref.
    """
    cache_key = "live_upcoming"
    cached = _get_cached(cache_key, max_age_hours=2)
    
    if cached is not None:
        print("📋 Usando fixtures de caché")
        df = pd.DataFrame(cached)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    
    print("📡 Descargando fixtures de FBref...")
    
    try:
        resp = requests.get(FBREF_SCHEDULE_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️ Error: {e}")
        return pd.DataFrame()
    
    tables = pd.read_html(resp.text)
    schedule_df = None
    
    for table in tables:
        cols = [str(c).lower() for c in table.columns.get_level_values(-1)]
        if "home" in cols and "away" in cols:
            schedule_df = table
            break
    
    if schedule_df is None:
        return pd.DataFrame()
    
    if isinstance(schedule_df.columns, pd.MultiIndex):
        schedule_df.columns = schedule_df.columns.get_level_values(-1)
    
    col_map = {}
    for c in schedule_df.columns:
        cl = str(c).lower().strip()
        if cl == "wk": col_map[c] = "fecha"
        elif cl == "date": col_map[c] = "date"
        elif cl == "home": col_map[c] = "home_team"
        elif cl == "away": col_map[c] = "away_team"
        elif cl == "score": col_map[c] = "score"
    
    schedule_df = schedule_df.rename(columns=col_map)
    
    # Filtrar partidos NO jugados
    if "score" in schedule_df.columns:
        upcoming = schedule_df[
            schedule_df["score"].isna() | 
            ~schedule_df["score"].astype(str).str.contains(r"\d+", regex=True)
        ].copy()
    else:
        upcoming = schedule_df.copy()
    
    keep = ["fecha", "date", "home_team", "away_team"]
    available = [c for c in keep if c in upcoming.columns]
    upcoming = upcoming[available].head(20).reset_index(drop=True)
    
    if "date" in upcoming.columns:
        upcoming["date"] = pd.to_datetime(upcoming["date"], errors="coerce")
    
    # Cache
    cache_data = upcoming.copy()
    if "date" in cache_data.columns:
        cache_data["date"] = cache_data["date"].astype(str)
    _save_cache(cache_key, cache_data.to_dict(orient="records"))
    
    return upcoming


def get_live_data():
    """
    Pipeline completo: standings, results, upcoming.
    Si el scraping falla, cae en datos locales.
    
    Returns: (standings_df, results_df, upcoming_df)
    """
    try:
        standings = scrape_standings()
        results = scrape_results()
        upcoming = scrape_upcoming()
        
        if standings.empty or results.empty:
            raise ValueError("Datos incompletos del scraping")
        
        print(f"\n✅ Datos live cargados:")
        print(f"   Tabla: {len(standings)} equipos")
        print(f"   Resultados: {len(results)} partidos")
        print(f"   Próximos: {len(upcoming)} partidos")
        
        return standings, results, upcoming
    
    except Exception as e:
        print(f"\n⚠️ Error scraping: {e}")
        print("   Usando datos locales como respaldo...")
        from src.data.real_data_2026 import (
            get_standings_df, get_results_df, get_upcoming_df
        )
        return get_standings_df(), get_results_df(), get_upcoming_df()


def is_live_available() -> bool:
    """Siempre disponible (scraping no necesita API key)."""
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  Live Scraper — Liga BetPlay")
    print("=" * 60)
    
    standings, results, upcoming = get_live_data()
    
    if not standings.empty:
        print("\n🏆 TABLA DE POSICIONES:")
        print(standings.to_string())
    
    if not results.empty:
        max_fecha = results["fecha"].max() if "fecha" in results.columns else "?"
        print(f"\n📊 RESULTADOS (hasta fecha {max_fecha}):")
        print(results.tail(10)[["fecha", "home_team", "home_goals", 
                                 "away_goals", "away_team"]].to_string())
