# 🚚 Flaschendepot - Service-Time Prediction MLOps

## 📋 Projektübersicht

Vollständiges **MLOps-System** für die **Service-Time-Vorhersage** bei Getränkelieferungen.  
Nutzt **1.5 Millionen echte Bestellungen** für Machine Learning Regression.

### 🎯 Ziel
Vorhersage der **Service-Zeit** (Minuten) basierend auf:
- 📦 Artikelanzahl & Gewicht
- 🏢 Stockwerk & Aufzug
- 🕐 Tageszeit & Wochentag  
- 📍 Warehouse & Kundentyp

---

## 📊 Daten - 4 Parquet-Dateien

| Datei | Zeilen | Beschreibung |
|-------|--------|-------------|
| **articles.parquet** | 15.6M | Artikel mit Gewichten |
| **orders.parquet** | 1.5M | Bestellinformationen |
| **driver_order_mapping.parquet** | 1.5M | Fahrer-Zuordnung |
| **service_times.parquet** ⭐ | 1.5M | **Service-Zeiten (Target)** |

### Zielvariable
**`service_time_in_minutes`**  
- Min: 0.02 min | Max: 360 min  
- **Median: 8.0 min** | Mean: 9.4 min  
- Regression-Problem

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

### 2. Datenverarbeitung

```powershell
python src\data\make_dataset.py
```

**Was passiert:**
- ✅ Lädt 4 Parquet-Dateien aus `data/raw/`
- ✅ Merged über `web_order_id`
- ✅ Aggregiert Artikel-Stats (Anzahl, Gewicht)
- ✅ Bereinigt & filtert Daten
- ✅ Train/Test Split (80/20)
- ✅ Speichert in `data/processed/`

**Output**: ~1.2M Training, ~307K Test

### 3. Exploratory Data Analysis

```powershell
jupyter notebook notebooks/01_eda_delivery_service.ipynb
```

**Notebook-Inhalte:**
- 📊 Service-Time Verteilungen
- 📈 Stockwerk vs Service-Zeit
- 🏋️ Gewicht & Artikelanzahl Impact
- 🕐 Zeitliche Muster-Analyse
- 💡 Key Insights

---

## 🏗️ Projekt-Architektur

```
flaschendepot/
├── data/
│   ├── raw/                    # 4 Parquet-Dateien hier!
│   │   ├── articles.parquet
│   │   ├── orders.parquet
│   │   ├── driver_order_mapping.parquet
│   │   └── service_times.parquet
│   └── processed/              # Train/Test CSVs
├── notebooks/
│   └── 01_eda_delivery_service.ipynb  # Explorative Analyse
├── src/
│   ├── data/
│   │   └── make_dataset.py     # Daten laden & mergen
│   ├── features/
│   │   └── build_features.py   # Feature Engineering
│   ├── models/
│   │   ├── train_model.py      # Training (Regression)
│   │   └── predict.py          # Vorhersagen
│   └── api/
│       └── main.py             # FastAPI REST API
├── configs/
│   └── config.yaml             # Konfiguration
├── models/                     # Trainierte Modelle
├── tests/                      # Unit Tests
└── scripts/
    └── train_pipeline.py       # End-to-End Pipeline
```

---

## 🔬 Features

### Input-Features

**Kategorisch:**
- `warehouse_id`: Warehouse-Standort
- `has_elevator`: Aufzug vorhanden? (boolean)
- `is_pre_order`: Vorbestellung? (boolean)
- `is_business`: B2B-Kunde? (boolean)

**Numerisch:**
- `floor`: Stockwerk (0-20+)
- `num_articles`: Anzahl Artikel
- `total_weight_g`: Gesamtgewicht in Gramm
- `avg_article_weight_g`: Durchschnittsgewicht
- `max_article_weight_g`: Maximales Artikel-Gewicht

**Zeitlich (aus Timestamps):**
- `hour_of_day`: Stunde (0-23)
- `day_of_week`: Wochentag (0-6)
- `is_weekend`: Wochenende? (0/1)
- `month`: Monat (1-12)

**Abgeleitete Features:**
- `total_weight_kg`: Gewicht in kg
- `difficulty_score`: Schwierigkeits-Score (Stockwerk + Gewicht + Aufzug)
- `order_size_category`: Größenkategorie (small/medium/large/very_large)

### Target Variable
- **`service_time_in_minutes`**: Service-Zeit in Minuten (Regression!)

---

## 🤖 Machine Learning

### Algorithmen
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

### Metriken
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)
- MAPE (Mean Absolute Percentage Error)

### Training

```powershell
# Komplette Pipeline
python scripts\train_pipeline.py

# Nur Training
python src\models\train_model.py
```

---

## 🌐 API (FastAPI)

### Server starten

```powershell
uvicorn src.api.main:app --reload
```

### Endpoints

**GET** `/health` - Health Check

**POST** `/predict` - Einzelvorhersage
```json
{
  "warehouse_id": 12,
  "has_elevator": false,
  "floor": 3.0,
  "is_pre_order": true,
  "is_business": false,
  "num_articles": 15,
  "total_weight_g": 25000,
  "hour_of_day": 14,
  "day_of_week": 2
}
```

**Response:**
```json
{
  "predicted_service_time": 11.5,
  "confidence_interval": [9.2, 13.8]
}
```

API Docs: http://localhost:8000/docs

---

## 🐳 Docker

```powershell
# Build
docker build -t flaschendepot:latest .

# Run Training
docker run --rm -v ${PWD}/data:/app/data flaschendepot

# Run API
docker-compose up api
```

---

## 📈 MLflow Tracking

```powershell
mlflow ui
```

Öffne: http://localhost:5000

Tracked automatisch:
- Hyperparameter
- Metriken (MAE, RMSE, R²)
- Modelle & Artefakte

---

## 🧪 Testing

```powershell
# Alle Tests
pytest

# Mit Coverage
pytest --cov=src --cov-report=html

# Spezifische Tests
pytest tests/test_data_processing.py
pytest tests/test_models.py
```

---

## 💡 Key Insights (aus EDA)

1. **Service-Zeit Durchschnitt**: ~9.4 Minuten (Median: 8.0 min)

2. **Stockwerk-Effekt**:
   - Pro Stockwerk: +0.3-0.5 min
   - Mit Aufzug: ~30% schneller

3. **Gewicht-Impact**:
   - Ab 30kg: Deutlicher Anstieg
   - Linear bis ~50kg, dann exponentiell

4. **Zeitliche Muster**:
   - Peak-Zeiten: 12-14 Uhr & 18-20 Uhr
   - Wochenende: +5-10% längere Service-Zeit

5. **Artikel-Anzahl**:
   - Moderater Einfluss
   - >20 Artikel: Signifikant länger

---

## 📚 Verwendung

### 1. Datenverarbeitung
```powershell
python src\data\make_dataset.py
```

### 2. Feature Engineering
```powershell
python src\features\build_features.py
```

### 3. Training
```powershell
python src\models\train_model.py
```

### 4. Vorhersagen
```python
from src.models.predict import DeliveryPredictor

predictor = DeliveryPredictor()

delivery_data = {
    'warehouse_id': 12,
    'floor': 3.0,
    'has_elevator': False,
    'num_articles': 15,
    'total_weight_g': 25000,
    'hour_of_day': 14
}

result = predictor.predict_single(delivery_data)
print(f"Geschätzte Service-Zeit: {result['prediction']:.1f} Minuten")
```

---

## 🔄 CI/CD Pipeline

GitHub Actions führt automatisch aus:
- ✅ Tests (Python 3.9, 3.10, 3.11)
- ✅ Linting (black, flake8, isort)
- ✅ Build & Package
- ✅ Docker Image Build

---

## 📦 DVC - Data Versioning

```powershell
# Daten tracken
dvc add data/raw/*.parquet
dvc add models/*.pkl

# Commit
git add data/.dvc models/.dvc
git commit -m "Track data and models"

# Push
dvc push
```

---

## 🎯 Projektziele & Use Cases

### Business Value
- ⏱️ **Bessere Routenplanung** durch genaue Zeitschätzungen
- 🚚 **Effizientere Tourenplanung** für Fahrer
- 📊 **Kapazitätsplanung** basierend auf erwarteten Service-Zeiten
- 💰 **Kosteneinsparungen** durch optimierte Routen

### Technical Goals
- ✅ Production-Ready MLOps Pipeline
- ✅ Reproduzierbare Experimente
- ✅ Automatisierte Tests & CI/CD
- ✅ API für Real-Time Predictions
- ✅ Versionierung (Code, Daten, Modelle)

---

## 👤 Autor

**Franz**  
Data Science MLOps Project

---

## 📝 Lizenz

MIT License

---

## 🙏 Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Data**: pandas, numpy, pyarrow (Parquet)
- **Visualization**: matplotlib, seaborn, plotly
- **MLOps**: MLflow, DVC
- **API**: FastAPI, uvicorn
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Containerization**: Docker, Docker Compose

---

**Happy Predicting! 🚀📦**
