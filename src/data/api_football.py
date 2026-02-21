"""
api_football.py — Conexión a API-Football para datos en vivo Liga BetPlay
Free tier: 100 requests/día en api-football.com (via RapidAPI o directo)

Para usar:
1. Regístrate gratis en https://www.api-football.com/
2. Copia tu API Key del dashboard
3. Pégala en data/api_key.txt o como variable de entorno API_FOOTBALL_KEY
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
API_KEY_FILE = BASE_DIR / "data" / "api_key.txt"
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Liga BetPlay IDs en API-Football v3
LEAGUE_ID = 239       # Primera A Colombia
COUNTRY = "Colombia"

# Tiempo de caché en horas (para no gastar requests innecesarios)
CACHE_HOURS = 4


def get_api_key() -> str:
    """Lee la API key desde archivo o variable de entorno."""
    import os
    
    # Primero intenta variable de entorno
    key = os.environ.get("API_FOOTBALL_KEY", "")
    if key:
        return key
    
    # Luego intenta archivo
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text().strip()
        if key:
            return key
    
    return ""


def _api_request(endpoint: str, params: dict = None) -> dict:
    """Hace una petición a la API-Football v3."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "❌ No se encontró API key.\n"
            "   Opción 1: Crea archivo data/api_key.txt con tu key\n"
            "   Opción 2: export API_FOOTBALL_KEY=tu_key"
        )
    
    url = f"https://v3.football.api-sports.io/{endpoint}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "v3.football.api-sports.io"
    }
    
    response = requests.get(url, headers=headers, params=params or {})
    response.raise_for_status()
    
    data = response.json()
    
    # Verificar errores de la API
    if data.get("errors"):
        errors = data["errors"]
        if isinstance(errors, dict) and errors:
            error_msg = list(errors.values())[0] if errors.values() else str(errors)
            raise ValueError(f"API Error: {error_msg}")
    
    return data


def _get_cached(cache_name: str, max_age_hours: int = CACHE_HOURS):
    """Lee datos del caché si existen y no han expirado."""
    cache_file = CACHE_DIR / f"{cache_name}.json"
    if cache_file.exists():
        mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mod_time < timedelta(hours=max_age_hours):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _save_cache(cache_name: str, data: dict):
    """Guarda datos en caché."""
    cache_file = CACHE_DIR / f"{cache_name}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_current_season() -> int:
    """Detecta la temporada actual."""
    now = datetime.now()
    return now.year


def fetch_standings(season: int = None) -> pd.DataFrame:
    """
    Obtiene la tabla de posiciones de la API.
    
    Returns: DataFrame con columnas: pos, team, pj, g, e, p, gf, gc, dif, pts
    """
    if season is None:
        season = get_current_season()
    
    cache_key = f"standings_{season}"
    cached = _get_cached(cache_key)
    
    if cached is None:
        print(f"📡 Descargando tabla de posiciones (temporada {season})...")
        data = _api_request("standings", {
            "league": LEAGUE_ID,
            "season": season
        })
        _save_cache(cache_key, data)
    else:
        print(f"📋 Usando tabla de caché (temporada {season})")
        data = cached
    
    # Parsear respuesta
    standings = []
    try:
        league_data = data["response"][0]["league"]["standings"]
        # Puede haber múltiples grupos (fase de grupos, etc.)
        for group in league_data:
            for team_data in group:
                standings.append({
                    "pos": team_data["rank"],
                    "team": team_data["team"]["name"],
                    "team_id": team_data["team"]["id"],
                    "logo": team_data["team"]["logo"],
                    "pj": team_data["all"]["played"],
                    "g": team_data["all"]["win"],
                    "e": team_data["all"]["draw"],
                    "p": team_data["all"]["lose"],
                    "gf": team_data["all"]["goals"]["for"],
                    "gc": team_data["all"]["goals"]["against"],
                    "dif": team_data["goalsDiff"],
                    "pts": team_data["points"],
                    "form": team_data.get("form", ""),
                })
    except (KeyError, IndexError) as e:
        print(f"⚠️ Error parseando standings: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(standings)
    
    # Guardar en processed
    df.to_csv(PROCESSED_DIR / "standings_live.csv", index=False, encoding="utf-8-sig")
    
    return df


def fetch_results(season: int = None) -> pd.DataFrame:
    """
    Obtiene todos los resultados de partidos jugados.
    
    Returns: DataFrame con columnas: fecha, date, home_team, away_team, 
             home_goals, away_goals, result, total_goals
    """
    if season is None:
        season = get_current_season()
    
    cache_key = f"results_{season}"
    cached = _get_cached(cache_key, max_age_hours=2)
    
    if cached is None:
        print(f"📡 Descargando resultados (temporada {season})...")
        data = _api_request("fixtures", {
            "league": LEAGUE_ID,
            "season": season,
            "status": "FT"  # Solo partidos terminados
        })
        _save_cache(cache_key, data)
    else:
        print(f"📋 Usando resultados de caché (temporada {season})")
        data = cached
    
    # Parsear respuesta
    results = []
    try:
        for fixture in data["response"]:
            match_date = fixture["fixture"]["date"][:10]
            round_str = fixture["league"].get("round", "")
            
            # Extraer número de fecha del round
            fecha = 0
            if "Regular Season" in round_str:
                try:
                    fecha = int(round_str.split(" - ")[-1])
                except (ValueError, IndexError):
                    pass
            
            home_goals = fixture["goals"]["home"]
            away_goals = fixture["goals"]["away"]
            
            if home_goals is None or away_goals is None:
                continue
            
            results.append({
                "fecha": fecha,
                "date": match_date,
                "home_team": fixture["teams"]["home"]["name"],
                "away_team": fixture["teams"]["away"]["name"],
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "home_logo": fixture["teams"]["home"]["logo"],
                "away_logo": fixture["teams"]["away"]["logo"],
            })
    except (KeyError, IndexError) as e:
        print(f"⚠️ Error parseando resultados: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["fecha", "date"]).reset_index(drop=True)
        
        # Calcular resultado
        df["result"] = np.where(
            df["home_goals"] > df["away_goals"], "H",
            np.where(df["home_goals"] == df["away_goals"], "D", "A")
        )
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        df["season"] = season
        df["phase"] = "Apertura"
        
        # Guardar
        df.to_csv(RAW_DIR / f"liga_betplay_{season}_api.csv", 
                  index=False, encoding="utf-8-sig")
        df.to_parquet(PROCESSED_DIR / f"matches_{season}_live.parquet", index=False)
    
    return df


def fetch_upcoming(season: int = None, next_n: int = 10) -> pd.DataFrame:
    """
    Obtiene los próximos partidos por jugar.
    """
    if season is None:
        season = get_current_season()
    
    cache_key = f"upcoming_{season}"
    cached = _get_cached(cache_key, max_age_hours=1)
    
    if cached is None:
        print(f"📡 Descargando próximos partidos...")
        data = _api_request("fixtures", {
            "league": LEAGUE_ID,
            "season": season,
            "status": "NS",  # Not Started
            "next": next_n
        })
        _save_cache(cache_key, data)
    else:
        print(f"📋 Usando fixtures del caché")
        data = cached
    
    fixtures = []
    try:
        for fixture in data["response"]:
            match_date = fixture["fixture"]["date"][:10]
            round_str = fixture["league"].get("round", "")
            
            fecha = 0
            if "Regular Season" in round_str:
                try:
                    fecha = int(round_str.split(" - ")[-1])
                except (ValueError, IndexError):
                    pass
            
            fixtures.append({
                "fecha": fecha,
                "date": match_date,
                "home_team": fixture["teams"]["home"]["name"],
                "away_team": fixture["teams"]["away"]["name"],
            })
    except (KeyError, IndexError) as e:
        print(f"⚠️ Error parseando fixtures: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(fixtures)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["fecha", "date"]).reset_index(drop=True)
    
    return df


def is_api_available() -> bool:
    """Verifica si la API key está configurada."""
    return bool(get_api_key())


def get_live_data():
    """
    Pipeline completo: obtiene standings, results, y upcoming.
    Retorna: (standings_df, results_df, upcoming_df)
    
    Si la API no está disponible, retorna datos locales.
    """
    if not is_api_available():
        print("⚠️ API key no configurada. Usando datos locales...")
        from src.data.real_data_2026 import (
            get_standings_df, get_results_df, get_upcoming_df
        )
        return get_standings_df(), get_results_df(), get_upcoming_df()
    
    try:
        standings = fetch_standings()
        results = fetch_results()
        upcoming = fetch_upcoming()
        
        print(f"✅ Datos live cargados:")
        print(f"   Tabla: {len(standings)} equipos")
        print(f"   Resultados: {len(results)} partidos")
        print(f"   Próximos: {len(upcoming)} partidos")
        
        return standings, results, upcoming
    except Exception as e:
        print(f"⚠️ Error cargando datos live: {e}")
        print("   Usando datos locales como respaldo...")
        from src.data.real_data_2026 import (
            get_standings_df, get_results_df, get_upcoming_df
        )
        return get_standings_df(), get_results_df(), get_upcoming_df()


if __name__ == "__main__":
    print("=" * 60)
    print("  API-Football — Liga BetPlay Connector")
    print("=" * 60)
    
    if not is_api_available():
        print("\n⚠️  API key no encontrada.")
        print("    Crea el archivo: data/api_key.txt")
        print("    Con tu API key de https://www.api-football.com/")
        print("\n    O usa: export API_FOOTBALL_KEY=tu_api_key")
    else:
        standings, results, upcoming = get_live_data()
        
        print("\n🏆 Tabla de Posiciones:")
        print(standings[["pos", "team", "pj", "pts"]].head(10).to_string())
        
        print(f"\n📊 Últimos resultados:")
        if not results.empty:
            print(results[["date", "home_team", "home_goals", "away_goals", 
                          "away_team"]].tail(5).to_string())
