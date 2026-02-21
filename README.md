# Proyecto Data Science & ML — Liga BetPlay Colombia ⚽🇨🇴

Proyecto de análisis de datos y Machine Learning enfocado en la **Liga BetPlay (Primera A de Colombia)**.

## 🎯 Objetivos
- Análisis exploratorio completo de la liga colombiana
- Predicción de resultados de partidos con ML
- Sistema de rating ELO dinámico
- Clustering de equipos por estilo de juego
- Dashboard interactivo con Streamlit

## 📁 Estructura
```
data futbol/
├── data/raw/           # Datos crudos
├── data/processed/     # Datos limpios (Parquet)
├── notebooks/          # Jupyter notebooks (EDA, modelos)
├── src/                # Código fuente reutilizable
├── dashboard/          # App Streamlit
├── models/             # Modelos entrenados (.pkl)
└── reports/            # Gráficos y reportes
```

## 🚀 Quickstart
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## 📊 Fuentes de Datos
- Kaggle: Primera A Colombia (2007-2022)
- FBref: Estadísticas por temporada
- API-Football: Data en tiempo real

## 🤖 Modelos
| Modelo | Objetivo | Algoritmos |
|--------|----------|------------|
| Predicción de Resultados | Win/Draw/Loss | XGBoost, LightGBM, Random Forest |
| ELO Rating | Ranking dinámico | ELO con factor K adaptativo |
| Clustering | Agrupar equipos | K-Means + PCA |
