"""
match_predictor.py — Modelo de predicción de resultados Liga BetPlay
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Features numéricas para el modelo
FEATURE_COLS = [
    "home_form_pts", "home_form_gf", "home_form_ga", "home_form_wins",
    "away_form_pts", "away_form_gf", "away_form_ga", "away_form_wins",
    "home_season_win_rate", "home_season_avg_gf", "home_season_avg_ga", "home_season_gd",
    "home_home_win_rate",
    "away_season_win_rate", "away_season_avg_gf", "away_season_avg_ga", "away_season_gd",
    "away_away_win_rate",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
]


def prepare_data(df: pd.DataFrame):
    """
    Prepara datos para entrenamiento.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    # Filtrar filas con suficiente historial
    df = df[df["home_form_pts"] > 0].copy()
    
    X = df[FEATURE_COLS].fillna(0)
    y = df["result"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le


def train_models(X_train, y_train, X_test, y_test, label_encoder) -> dict:
    """
    Entrena múltiples modelos y compara rendimiento.
    
    Returns:
        Dict con nombre → (modelo, accuracy, report)
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42,
                                                   multi_class="multinomial"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 random_state=42, n_jobs=-1),
    }
    
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False, 
            eval_metric="mlogloss", verbosity=0
        )
    
    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    
    results = {}
    best_accuracy = 0
    best_model_name = None
    
    for name, model in models.items():
        print(f"\n🔄 Entrenando {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        report_str = classification_report(
            y_test, y_pred, 
            target_names=label_encoder.classes_
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "report": report,
            "report_str": report_str,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
    
    print(f"\n🏆 Mejor modelo: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # Guardar mejor modelo
    best = results[best_model_name]["model"]
    model_path = MODELS_DIR / "best_match_predictor.pkl"
    joblib.dump(best, model_path)
    
    le_path = MODELS_DIR / "label_encoder.pkl"
    joblib.dump(label_encoder, le_path)
    
    print(f"Modelo guardado: {model_path}")
    
    return results, best_model_name


def predict_match(home_features: dict, away_features: dict,
                   model_path: str = None) -> dict:
    """
    Predice el resultado de un partido usando el modelo entrenado.
    
    Args:
        home_features: Dict con features del equipo local
        away_features: Dict con features del equipo visitante
        model_path: Ruta al modelo guardado
    
    Returns:
        Dict con probabilidades
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_match_predictor.pkl"
    
    model = joblib.load(model_path)
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    
    # Construir feature vector
    features = {}
    for col in FEATURE_COLS:
        if col.startswith("home_"):
            key = col.replace("home_", "")
            features[col] = home_features.get(key, 0)
        elif col.startswith("away_"):
            key = col.replace("away_", "")
            features[col] = away_features.get(key, 0)
        else:
            features[col] = home_features.get(col, away_features.get(col, 0))
    
    X = pd.DataFrame([features])[FEATURE_COLS]
    
    proba = model.predict_proba(X)[0]
    classes = le.inverse_transform(range(len(proba)))
    
    return dict(zip(classes, proba.round(3)))


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Obtiene importancia de features del modelo."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame()
    
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    return fi


if __name__ == "__main__":
    print("=" * 60)
    print("  Match Predictor — Liga BetPlay")
    print("=" * 60)
    
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("❌ Ejecuta primero feature_engineering.py para generar features")
        exit(1)
    
    df = pd.read_parquet(features_path)
    print(f"Datos cargados: {df.shape}")
    
    X_train, X_test, y_train, y_test, le = prepare_data(df)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results, best_name = train_models(X_train, y_train, X_test, y_test, le)
    
    print(f"\n📊 Reporte del mejor modelo ({best_name}):")
    print(results[best_name]["report_str"])
