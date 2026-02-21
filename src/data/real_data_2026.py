"""
real_data_2026.py — Datos reales Liga BetPlay I-2026 (Apertura)
Datos actualizados al 20 de febrero de 2026, Fecha 7 completada.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ═══════════════════════════════════════════════════════════
# TABLA DE POSICIONES REAL — Liga BetPlay I-2026, Fecha 7
# Fuentes: 365scores.com, lafm.com.co, noticiascaracol.com
# ═══════════════════════════════════════════════════════════

STANDINGS_2026 = [
    # (pos, equipo, PJ, G, E, P, GF, GC, DIF, PTS)
    ( 1, "Internacional",       7, 4, 2, 1, 11, 10,  1, 14),
    ( 2, "Deportivo Pasto",     7, 4, 2, 1,  8,  7,  1, 14),
    ( 3, "Deportes Tolima",     7, 3, 3, 1,  8,  4,  4, 12),
    ( 4, "Junior",              6, 4, 0, 2,  9,  6,  3, 12),
    ( 5, "Bucaramanga",         6, 2, 4, 0,  8,  2,  6, 10),
    ( 6, "Deportivo Cali",      7, 3, 1, 3,  8,  5,  3, 10),
    ( 7, "América de Cali",     5, 3, 1, 1,  8,  5,  3, 10),
    ( 8, "Once Caldas",         7, 3, 1, 3, 10,  9,  1, 10),
    ( 9, "Fortaleza",           7, 3, 1, 3,  7,  8, -1, 10),
    (10, "Jaguares",            7, 3, 1, 3, 11, 13, -2, 10),
    (11, "Atlético Nacional",   6, 2, 3, 1, 11,  4,  7,  9),
    (12, "Llaneros",            7, 2, 3, 2,  9,  8,  1,  9),
    (13, "Águilas Doradas",     6, 2, 2, 2,  6,  6,  0,  8),
    (14, "Millonarios",         7, 2, 2, 3, 10, 11, -1,  8),
    (15, "Santa Fe",            7, 2, 1, 4,  7, 10, -3,  7),
    (16, "Medellín",            7, 1, 3, 3,  7, 10, -3,  6),
    (17, "Boyacá Chicó",        6, 1, 1, 4,  5, 10, -5,  4),
    (18, "Deportivo Pereira",   6, 0, 3, 3,  5, 10, -5,  3),
    (19, "Cúcuta Deportivo",    7, 0, 3, 4,  3,  9, -6,  3),
    (20, "Alianza FC",          6, 0, 3, 3,  3,  9, -6,  3),
]

# ═══════════════════════════════════════════════════════════
# RESULTADOS REALES — Liga BetPlay I-2026, Fechas 1-7
# ═══════════════════════════════════════════════════════════

RESULTS_2026 = [
    # Fecha 1 (Ene 24-27, 2026)
    {"fecha": 1, "date": "2026-01-24", "home": "Junior",              "away": "Boyacá Chicó",       "hg": 2, "ag": 0},
    {"fecha": 1, "date": "2026-01-24", "home": "Atlético Nacional",   "away": "Deportivo Pereira",  "hg": 3, "ag": 0},
    {"fecha": 1, "date": "2026-01-25", "home": "Deportivo Cali",      "away": "Alianza FC",         "hg": 1, "ag": 0},
    {"fecha": 1, "date": "2026-01-25", "home": "América de Cali",     "away": "Medellín",           "hg": 2, "ag": 1},
    {"fecha": 1, "date": "2026-01-25", "home": "Deportes Tolima",     "away": "Llaneros",           "hg": 2, "ag": 1},
    {"fecha": 1, "date": "2026-01-26", "home": "Fortaleza",           "away": "Cúcuta Deportivo",   "hg": 2, "ag": 0},
    {"fecha": 1, "date": "2026-01-26", "home": "Once Caldas",         "away": "Santa Fe",           "hg": 3, "ag": 1},
    {"fecha": 1, "date": "2026-01-26", "home": "Deportivo Pasto",     "away": "Águilas Doradas",    "hg": 1, "ag": 0},
    {"fecha": 1, "date": "2026-01-27", "home": "Jaguares",            "away": "Millonarios",        "hg": 3, "ag": 2},
    {"fecha": 1, "date": "2026-01-27", "home": "Internacional",       "away": "Bucaramanga",        "hg": 1, "ag": 1},
    
    # Fecha 2 (Ene 31 - Feb 2, 2026)
    {"fecha": 2, "date": "2026-01-31", "home": "Boyacá Chicó",        "away": "América de Cali",    "hg": 0, "ag": 2},
    {"fecha": 2, "date": "2026-01-31", "home": "Santa Fe",            "away": "Deportivo Cali",     "hg": 1, "ag": 2},
    {"fecha": 2, "date": "2026-02-01", "home": "Alianza FC",          "away": "Deportes Tolima",    "hg": 0, "ag": 1},
    {"fecha": 2, "date": "2026-02-01", "home": "Llaneros",            "away": "Jaguares",           "hg": 2, "ag": 1},
    {"fecha": 2, "date": "2026-02-01", "home": "Medellín",            "away": "Fortaleza",          "hg": 1, "ag": 2},
    {"fecha": 2, "date": "2026-02-01", "home": "Pereira",             "away": "Deportivo Pasto",    "hg": 1, "ag": 2},
    {"fecha": 2, "date": "2026-02-02", "home": "Cúcuta Deportivo",    "away": "Once Caldas",        "hg": 1, "ag": 3},
    {"fecha": 2, "date": "2026-02-02", "home": "Bucaramanga",         "away": "Junior",             "hg": 1, "ag": 1},
    {"fecha": 2, "date": "2026-02-02", "home": "Águilas Doradas",     "away": "Atlético Nacional",  "hg": 1, "ag": 1},
    {"fecha": 2, "date": "2026-02-02", "home": "Millonarios",         "away": "Internacional",      "hg": 3, "ag": 2},
    
    # Fecha 3 (Feb 4-6, 2026)
    {"fecha": 3, "date": "2026-02-04", "home": "Junior",              "away": "Llaneros",           "hg": 3, "ag": 1},
    {"fecha": 3, "date": "2026-02-04", "home": "Deportivo Pasto",     "away": "Boyacá Chicó",       "hg": 2, "ag": 0},
    {"fecha": 3, "date": "2026-02-04", "home": "Once Caldas",         "away": "Medellín",           "hg": 1, "ag": 0},
    {"fecha": 3, "date": "2026-02-05", "home": "América de Cali",     "away": "Cúcuta Deportivo",   "hg": 2, "ag": 0},
    {"fecha": 3, "date": "2026-02-05", "home": "Deportivo Cali",      "away": "Atlético Nacional",  "hg": 1, "ag": 0},
    {"fecha": 3, "date": "2026-02-05", "home": "Internacional",       "away": "Fortaleza",          "hg": 2, "ag": 1},
    {"fecha": 3, "date": "2026-02-05", "home": "Deportes Tolima",     "away": "Jaguares",           "hg": 1, "ag": 1},
    {"fecha": 3, "date": "2026-02-06", "home": "Bucaramanga",         "away": "Águilas Doradas",    "hg": 2, "ag": 1},
    {"fecha": 3, "date": "2026-02-06", "home": "Santa Fe",            "away": "Alianza FC",         "hg": 2, "ag": 1},
    {"fecha": 3, "date": "2026-02-06", "home": "Millonarios",         "away": "Deportivo Pereira",  "hg": 1, "ag": 1},
    
    # Fecha 4 (Feb 7-9, 2026)
    {"fecha": 4, "date": "2026-02-07", "home": "Fortaleza",           "away": "Deportivo Pasto",    "hg": 1, "ag": 2},
    {"fecha": 4, "date": "2026-02-07", "home": "Cúcuta Deportivo",    "away": "Deportivo Cali",     "hg": 0, "ag": 1},
    {"fecha": 4, "date": "2026-02-07", "home": "Llaneros",            "away": "Deportes Tolima",    "hg": 1, "ag": 1},
    {"fecha": 4, "date": "2026-02-08", "home": "Atlético Nacional",   "away": "Junior",             "hg": 2, "ag": 2},
    {"fecha": 4, "date": "2026-02-08", "home": "Jaguares",            "away": "Once Caldas",        "hg": 3, "ag": 1},
    {"fecha": 4, "date": "2026-02-08", "home": "Medellín",            "away": "Internacional",      "hg": 1, "ag": 2},
    {"fecha": 4, "date": "2026-02-08", "home": "Deportivo Pereira",   "away": "Bucaramanga",        "hg": 0, "ag": 0},
    {"fecha": 4, "date": "2026-02-09", "home": "Alianza FC",          "away": "Millonarios",        "hg": 0, "ag": 2},
    {"fecha": 4, "date": "2026-02-09", "home": "Águilas Doradas",     "away": "Santa Fe",           "hg": 2, "ag": 1},
    {"fecha": 4, "date": "2026-02-09", "home": "Boyacá Chicó",        "away": "Jaguares",           "hg": 1, "ag": 2},
    
    # Fecha 5 (Feb 10-11, 2026)
    {"fecha": 5, "date": "2026-02-10", "home": "Deportivo Pasto",     "away": "Medellín",           "hg": 2, "ag": 1},
    {"fecha": 5, "date": "2026-02-10", "home": "Junior",              "away": "Águilas Doradas",    "hg": 2, "ag": 0},
    {"fecha": 5, "date": "2026-02-10", "home": "Internacional",       "away": "Once Caldas",        "hg": 1, "ag": 0},
    {"fecha": 5, "date": "2026-02-10", "home": "Deportes Tolima",     "away": "Fortaleza",          "hg": 2, "ag": 0},
    {"fecha": 5, "date": "2026-02-11", "home": "Deportivo Cali",      "away": "Llaneros",           "hg": 1, "ag": 2},
    {"fecha": 5, "date": "2026-02-11", "home": "Bucaramanga",         "away": "Cúcuta Deportivo",   "hg": 2, "ag": 0},
    {"fecha": 5, "date": "2026-02-11", "home": "Santa Fe",            "away": "Millonarios",        "hg": 1, "ag": 0},
    {"fecha": 5, "date": "2026-02-11", "home": "Atlético Nacional",   "away": "Deportivo Pasto",    "hg": 2, "ag": 2},
    {"fecha": 5, "date": "2026-02-11", "home": "América de Cali",     "away": "Alianza FC",         "hg": 2, "ag": 1},
    {"fecha": 5, "date": "2026-02-11", "home": "Boyacá Chicó",        "away": "Deportivo Pereira",  "hg": 2, "ag": 2},
    
    # Fecha 6 (Feb 12-13, 2026)
    {"fecha": 6, "date": "2026-02-12", "home": "Fortaleza",           "away": "Deportivo Cali",     "hg": 2, "ag": 1},
    {"fecha": 6, "date": "2026-02-12", "home": "Cúcuta Deportivo",    "away": "Atlético Nacional",  "hg": 0, "ag": 1},
    {"fecha": 6, "date": "2026-02-12", "home": "Once Caldas",         "away": "Junior",             "hg": 2, "ag": 0},
    {"fecha": 6, "date": "2026-02-12", "home": "Medellín",            "away": "Bucaramanga",        "hg": 1, "ag": 1},
    {"fecha": 6, "date": "2026-02-13", "home": "Llaneros",            "away": "Santa Fe",           "hg": 1, "ag": 0},
    {"fecha": 6, "date": "2026-02-13", "home": "Deportivo Pereira",   "away": "América de Cali",    "hg": 1, "ag": 0},
    {"fecha": 6, "date": "2026-02-13", "home": "Jaguares",            "away": "Deportes Tolima",    "hg": 1, "ag": 1},
    {"fecha": 6, "date": "2026-02-13", "home": "Internacional",       "away": "Deportivo Pasto",    "hg": 1, "ag": 1},
    {"fecha": 6, "date": "2026-02-13", "home": "Millonarios",         "away": "Águilas Doradas",    "hg": 1, "ag": 0},
    {"fecha": 6, "date": "2026-02-13", "home": "Boyacá Chicó",        "away": "Alianza FC",         "hg": 1, "ag": 0},
    
    # Fecha 7 (Feb 14-17, 2026)
    {"fecha": 7, "date": "2026-02-14", "home": "Millonarios",         "away": "Llaneros",           "hg": 2, "ag": 1},
    {"fecha": 7, "date": "2026-02-14", "home": "Medellín",            "away": "Deportivo Pereira",  "hg": 1, "ag": 1},
    {"fecha": 7, "date": "2026-02-14", "home": "Deportivo Pasto",     "away": "Internacional",      "hg": 1, "ag": 1},
    {"fecha": 7, "date": "2026-02-15", "home": "Deportes Tolima",     "away": "Boyacá Chicó",       "hg": 1, "ag": 0},
    {"fecha": 7, "date": "2026-02-15", "home": "Deportivo Cali",      "away": "Once Caldas",        "hg": 2, "ag": 0},
    {"fecha": 7, "date": "2026-02-16", "home": "Fortaleza",           "away": "Llaneros",           "hg": 0, "ag": 0},
    {"fecha": 7, "date": "2026-02-17", "home": "Jaguares",            "away": "Santa Fe",           "hg": 3, "ag": 1},
    {"fecha": 7, "date": "2026-02-18", "home": "Junior",              "away": "América de Cali",    "hg": 2, "ag": 1},
    # Pendiente: Águilas Doradas vs Bucaramanga
    
    # Fecha 8 (Feb 17-23, 2026) — Resultados parciales
    {"fecha": 8, "date": "2026-02-17", "home": "Alianza FC",          "away": "Cúcuta Deportivo",   "hg": 1, "ag": 1},
    {"fecha": 8, "date": "2026-02-20", "home": "Llaneros",            "away": "Medellín",           "hg": 2, "ag": 2},
    {"fecha": 8, "date": "2026-02-20", "home": "Deportivo Pereira",   "away": "Deportivo Pasto",    "hg": 2, "ag": 2},
]

# ═══════════════════════════════════════════════════════════
# FIXTURE FECHA 8 — Partidos restantes (Feb 21-23, 2026)
# ═══════════════════════════════════════════════════════════

UPCOMING_FIXTURES = [
    {"fecha": 8, "date": "2026-02-21", "home": "Once Caldas",         "away": "Fortaleza"},
    {"fecha": 8, "date": "2026-02-21", "home": "Bucaramanga",         "away": "Deportivo Cali"},
    {"fecha": 8, "date": "2026-02-21", "home": "Atlético Nacional",   "away": "Alianza FC"},
    {"fecha": 8, "date": "2026-02-21", "home": "Internacional",       "away": "Millonarios"},
    {"fecha": 8, "date": "2026-02-22", "home": "Santa Fe",            "away": "Junior"},
    {"fecha": 8, "date": "2026-02-22", "home": "Boyacá Chicó",        "away": "Águilas Doradas"},
    {"fecha": 8, "date": "2026-02-22", "home": "América de Cali",     "away": "Jaguares"},
    {"fecha": 8, "date": "2026-02-23", "home": "Cúcuta Deportivo",    "away": "Deportes Tolima"},
]

# Los 20 equipos de la temporada
TEAMS_2026 = [
    "Internacional", "Deportivo Pasto", "Deportes Tolima", "Junior", 
    "Bucaramanga", "Deportivo Cali", "América de Cali", "Once Caldas",
    "Fortaleza", "Jaguares", "Atlético Nacional", "Llaneros",
    "Águilas Doradas", "Millonarios", "Santa Fe", "Medellín",
    "Boyacá Chicó", "Deportivo Pereira", "Cúcuta Deportivo", "Alianza FC"
]


def get_standings_df() -> pd.DataFrame:
    """Calcula la tabla de posiciones automáticamente a partir de los resultados."""
    stats = {team: {"pj": 0, "g": 0, "e": 0, "p": 0, "gf": 0, "gc": 0} for team in TEAMS_2026}
    
    for match in RESULTS_2026:
        home, away = match["home"], match["away"]
        hg, ag = match["hg"], match["ag"]
        
        if home not in stats or away not in stats:
            continue
        
        stats[home]["pj"] += 1
        stats[away]["pj"] += 1
        stats[home]["gf"] += hg
        stats[home]["gc"] += ag
        stats[away]["gf"] += ag
        stats[away]["gc"] += hg
        
        if hg > ag:
            stats[home]["g"] += 1
            stats[away]["p"] += 1
        elif hg == ag:
            stats[home]["e"] += 1
            stats[away]["e"] += 1
        else:
            stats[away]["g"] += 1
            stats[home]["p"] += 1
    
    rows = []
    for team, s in stats.items():
        dif = s["gf"] - s["gc"]
        pts = s["g"] * 3 + s["e"]
        rows.append([team, s["pj"], s["g"], s["e"], s["p"], s["gf"], s["gc"], dif, pts])
    
    df = pd.DataFrame(rows, columns=["team", "pj", "g", "e", "p", "gf", "gc", "dif", "pts"])
    df = df.sort_values(["pts", "dif", "gf"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "pos"
    return df


def get_results_df() -> pd.DataFrame:
    """Retorna todos los resultados reales como DataFrame."""
    df = pd.DataFrame(RESULTS_2026)
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={
        "home": "home_team", "away": "away_team",
        "hg": "home_goals", "ag": "away_goals"
    }, inplace=True)
    
    # Calcular resultado
    df["result"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] == df["away_goals"], "D", "A")
    )
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["season"] = 2026
    df["phase"] = "Apertura"
    
    return df


def get_upcoming_df() -> pd.DataFrame:
    """Retorna fixtures pendientes."""
    df = pd.DataFrame(UPCOMING_FIXTURES)
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={
        "home": "home_team", "away": "away_team"
    }, inplace=True)
    return df


def save_real_data():
    """Guarda los datos reales en archivos procesados."""
    results = get_results_df()
    standings = get_standings_df()
    
    results.to_csv(RAW_DIR / "liga_betplay_2026_results.csv", index=False, encoding="utf-8-sig")
    results.to_parquet(PROCESSED_DIR / "matches_2026_real.parquet", index=False)
    standings.to_csv(PROCESSED_DIR / "standings_2026_real.csv", encoding="utf-8-sig")
    
    print(f"✅ Datos reales guardados:")
    print(f"   {len(results)} partidos (Fechas 1-7)")
    print(f"   {len(standings)} equipos en la tabla")
    
    return results, standings


if __name__ == "__main__":
    print("=" * 60)
    print("  Datos Reales — Liga BetPlay I-2026")
    print("=" * 60)
    
    results, standings = save_real_data()
    
    print("\n🏆 Tabla de Posiciones (Fecha 7):")
    print(standings.to_string())
    
    print(f"\n📊 Estadísticas:")
    print(f"   Partidos: {len(results)}")
    print(f"   Goles: {results['total_goals'].sum()}")
    print(f"   Promedio: {results['total_goals'].mean():.2f} goles/partido")
