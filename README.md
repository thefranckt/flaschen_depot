# Service Time Prediction - MLOps Projekt

Ein End-to-End Machine Learning System zur Vorhersage von Lieferzeiten (Service Time) für Flaschenpost-Bestellungen.

## 📋 Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Projektstruktur](#projektstruktur)
- [Setup](#setup)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [API](#api)
- [Logging](#logging)
- [Reproduzierbarkeit](#reproduzierbarkeit)
- [Modellversionierung](#modellversionierung)

---

## 🎯 Projektübersicht

Dieses Projekt implementiert eine vollständige MLOps-Pipeline zur Vorhersage der Service Time (Dauer der Auslieferung beim Kunden) basierend auf historischen Bestelldaten.

**Hauptkomponenten:**
- 📊 Explorative Datenanalyse (EDA)
- 🔧 Feature Engineering Pipeline
- 🤖 Model Training mit MLflow Tracking
- 🚀 REST API für Predictions
- 📝 Feature und Prediction Logging
- 🔄 Modellversionierung

**Technologie-Stack:**
- Python 3.13
- scikit-learn, XGBoost, LightGBM
- FastAPI
- MLflow
- Pandas, NumPy
- SQLite (für Logging)

---

## 📁 Projektstruktur

```
flaschendepot/
├── config/
│   └── config.yaml              # Zentrale Konfiguration
├── data/
│   ├── raw/                     # Rohdaten (Parquet-Dateien)
│   │   ├── orders.parquet
│   │   ├── articles.parquet
│   │   ├── service_times.parquet
│   │   └── driver_order_mapping.parquet
│   └── processed/               # Verarbeitete Daten
│       ├── features.parquet
│       ├── target.parquet
│       └── full_dataset.parquet
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # EDA Notebook
├── plots/                       # Generierte Visualisierungen (aus EDA)
│   ├── README.md
│   ├── .gitkeep
│   └── *.png                    # Alle EDA Plots (ignoriert von Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Daten laden
│   ├── feature_engineering.py   # Feature Engineering
│   └── logger.py                # Feature & Prediction Logging
├── models/                      # Gespeicherte Modelle
│   ├── model_latest.joblib
│   └── model_*.joblib
├── logs/                        # Log-Datenbanken
│   ├── feature_store.db
│   ├── prediction_store.db
│   └── app.log
├── mlruns/                      # MLflow Tracking
├── tests/                       # Unit Tests
├── train.py                     # Training Script
├── api.py                       # FastAPI Application
├── requirements.txt             # Python Dependencies
├── .env.example                 # Environment Template
├── .gitignore
└── README.md
```

---

## 🚀 Setup

### 1. Voraussetzungen

- Python 3.13+
- Git
- Virtual Environment Tool (venv)

### 2. Repository klonen

```bash
git clone <repository-url>
cd flaschendepot
```

### 3. Virtual Environment erstellen

```bash
# Windows (PowerShell/CMD)
python -m venv myenv
myenv\Scripts\activate

# Windows (Git Bash)
python -m venv myenv
source myenv/Scripts/activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

### 4. Dependencies installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Daten vorbereiten

Legen Sie die Parquet-Dateien in den `data/raw/` Ordner:
- `orders.parquet`
- `articles.parquet`
- `service_times.parquet`
- `driver_order_mapping.parquet`

### 6. Konfiguration (optional)

Kopieren Sie `.env.example` zu `.env` und passen Sie die Werte an:

```bash
cp .env.example .env
```

Bearbeiten Sie `config/config.yaml` für erweiterte Konfiguration.

---

## 🔧 Feature Engineering

### Datenpipeline

Die Feature Engineering Pipeline führt folgende Schritte aus:

1. **Daten Laden**
   - Orders: Bestellinformationen (Etage, Aufzug, Geschäftskunde)
   - Articles: Artikelinformationen (Gewicht, Anzahl)
   - Service Times: Zielvariable (service_time_in_minutes)
   - Driver Mapping: Zuordnung Fahrer-Bestellungen

2. **Daten Zusammenführen**
   - Join auf `web_order_id`
   - Aggregation der Artikel-Daten pro Bestellung

3. **Datenbereinigung**
   - Entfernung von Missing Values in Zielvariable
   - Entfernung negativer/null Service Times
   - Imputation von Missing Values in Features
   - Ausreißer-Erkennung mit IQR-Methode

4. **Feature Engineering**
   - **Zeitliche Features:**
     - `order_hour`: Stunde der Bestellung
     - `order_day_of_week`: Wochentag (0=Montag)
     - `order_month`: Monat
     - `is_weekend`: Wochenende-Indikator
   
   - **Aggregierte Features:**
     - `total_boxes`: Anzahl Kisten
     - `total_articles`: Anzahl Artikel
     - `total_weight_g`: Gesamtgewicht
     - `avg_article_weight_g`: Durchschnittsgewicht pro Artikel
     - `max_article_weight_g`: Maximales Artikelgewicht
     - `min_article_weight_g`: Minimales Artikelgewicht
     - `weight_per_box`: Gewicht pro Kiste
   
   - **Interaktions-Features:**
     - `floor_elevator_interaction`: Etage × Aufzug
     - `business_floor_interaction`: Geschäftskunde × Etage

### EDA durchführen

```bash
# Jupyter Notebook starten
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

Das Notebook enthält:
- Datenqualitätsprüfung
- Deskriptive Statistiken
- Verteilungsanalysen
- Korrelationsanalyse
- Feature-Wichtigkeit Visualisierungen

### Programmatische Verwendung

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer

# Daten laden
loader = DataLoader("data/raw")
orders, articles, service_times, driver_mapping = loader.load_all()

# Feature Engineering
engineer = FeatureEngineer(random_state=42)
X, y, feature_names, df = engineer.process_pipeline(
    orders, articles, service_times, driver_mapping
)
```

---

## 🤖 Model Training

### Training ausführen

```bash
python train.py
```

### Was passiert beim Training?

1. **Daten laden und vorbereiten**
   - Lädt Rohdaten
   - Führt Feature Engineering durch
   - Speichert verarbeitete Daten in `data/processed/`

2. **Daten splitten**
   - Training: 70%
   - Validation: 10%
   - Test: 20%
   - Stratifiziert mit `random_state=42`

3. **Modell trainieren**
   - Standardmäßig: Random Forest Regressor
   - Konfigurierbar über `config/config.yaml`
   - Unterstützte Modelle: Random Forest, XGBoost, LightGBM

4. **Evaluierung**
   - Metriken: RMSE, MAE, R²
   - Auf Training, Validation und Test Set
   - Feature Importance Analyse

5. **Speicherung**
   - Modell als `models/model_latest.joblib`
   - Zeitstempel-Version: `models/model_{type}_{timestamp}.joblib`
   - Metadata als YAML

6. **MLflow Tracking**
   - Alle Parameter und Metriken werden geloggt
   - Modell wird registriert
   - Artefakte werden gespeichert

### Modell konfigurieren

Bearbeiten Sie `config/config.yaml`:

```yaml
model:
  type: "random_forest"  # Options: random_forest, xgboost, lightgbm
  random_state: 42
  
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5
```

### MLflow UI ansehen

```bash
mlflow ui
```

Öffnen Sie http://localhost:5000 im Browser.

---

## 🚀 API

### API starten

```bash
# Direkt
python api.py

# Oder mit uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Die API ist verfügbar unter: http://localhost:8000

### API Dokumentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints

#### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-11-26T10:30:00",
  "models_loaded": 1,
  "available_versions": ["latest"]
}
```

#### 2. Single Prediction

```bash
POST /predict?model_version=latest
Content-Type: application/json

{
  "driver_id": "D001",
  "web_order_id": "ORDER_12345"
}
```

**Response:**
```json
{
  "driver_id": "D001",
  "web_order_id": "ORDER_12345",
  "predicted_service_time": 12.5,
  "model_version": "latest",
  "timestamp": "2024-11-26T10:30:00",
  "request_id": "uuid-1234"
}
```

#### 3. Batch Predictions

```bash
POST /predict/batch
Content-Type: application/json

{
  "model_version": "latest",
  "requests": [
    {"driver_id": "D001", "web_order_id": "ORDER_1"},
    {"driver_id": "D002", "web_order_id": "ORDER_2"},
    {"driver_id": "D003", "web_order_id": "ORDER_3"}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "driver_id": "D001",
      "web_order_id": "ORDER_1",
      "predicted_service_time": 12.5,
      "model_version": "latest",
      "timestamp": "2024-11-26T10:30:00",
      "request_id": "batch-uuid"
    },
    ...
  ],
  "total_count": 3,
  "request_id": "batch-uuid",
  "timestamp": "2024-11-26T10:30:00"
}
```

#### 4. List Models

```bash
GET /models
```

#### 5. Get Feature Logs

```bash
GET /logs/features?web_order_id=ORDER_1&limit=10
```

#### 6. Get Prediction Logs

```bash
GET /logs/predictions?driver_id=D001&limit=10
```

#### 7. Get Statistics

```bash
GET /logs/statistics
```

### API Beispiele (Python)

```python
import requests

# Single Prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "driver_id": "D001",
        "web_order_id": "ORDER_12345"
    },
    params={"model_version": "latest"}
)
print(response.json())

# Batch Prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "model_version": "latest",
        "requests": [
            {"driver_id": "D001", "web_order_id": "ORDER_1"},
            {"driver_id": "D002", "web_order_id": "ORDER_2"}
        ]
    }
)
print(response.json())
```

### API Beispiele (cURL)

```bash
# Single Prediction
curl -X POST "http://localhost:8000/predict?model_version=latest" \
  -H "Content-Type: application/json" \
  -d '{"driver_id": "D001", "web_order_id": "ORDER_12345"}'

# Batch Prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "latest",
    "requests": [
      {"driver_id": "D001", "web_order_id": "ORDER_1"},
      {"driver_id": "D002", "web_order_id": "ORDER_2"}
    ]
  }'
```

---

## 📝 Logging

### Logging-System

Das Projekt implementiert zwei separate Logging-Datenbanken:

1. **Feature Store** (`logs/feature_store.db`)
   - Speichert alle Features für jede Prediction
   - Ermöglicht Feature-Lookup und Debugging

2. **Prediction Store** (`logs/prediction_store.db`)
   - Speichert alle Predictions mit Metadata
   - Tracking von Modellversionen
   - Request-IDs für Batch-Requests

### Logging-Schema

**Feature Logs:**
```sql
CREATE TABLE feature_logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    web_order_id TEXT,
    driver_id TEXT,
    features TEXT (JSON),
    model_version TEXT
)
```

**Prediction Logs:**
```sql
CREATE TABLE prediction_logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    web_order_id TEXT,
    driver_id TEXT,
    predicted_service_time REAL,
    model_version TEXT,
    request_id TEXT
)
```

### Logs inspizieren

#### Über API

```bash
# Feature Logs abrufen
curl "http://localhost:8000/logs/features?limit=10"

# Prediction Logs abrufen
curl "http://localhost:8000/logs/predictions?limit=10"

# Statistiken
curl "http://localhost:8000/logs/statistics"
```

#### Programmatisch

```python
from src.logger import FeatureLogger, PredictionLogger

# Feature Logs
feature_logger = FeatureLogger("logs/feature_store.db")
logs = feature_logger.get_features(web_order_id="ORDER_1", limit=10)
print(logs)

# Prediction Logs
prediction_logger = PredictionLogger("logs/prediction_store.db")
logs = prediction_logger.get_predictions(driver_id="D001", limit=10)
print(logs)

# Statistiken
stats = prediction_logger.get_statistics()
print(stats)
```

#### Mit SQLite

```bash
sqlite3 logs/feature_store.db "SELECT * FROM feature_logs LIMIT 10;"
sqlite3 logs/prediction_store.db "SELECT * FROM prediction_logs LIMIT 10;"
```

### Log-Retention

Konfigurierbar in `config/config.yaml`:

```yaml
logging:
  max_log_age_days: 90  # Logs älter als 90 Tage werden gelöscht
```

---

## 🔄 Reproduzierbarkeit

Das Projekt implementiert mehrere Mechanismen zur Sicherstellung reproduzierbarer Ergebnisse:

### 1. Random Seeds

Alle zufälligen Operationen verwenden denselben Seed:

```yaml
# config/config.yaml
model:
  random_state: 42

reproducibility:
  seed: 42
  deterministic: true
```

**Verwendung:**
- Train-Test Split
- Model Initialization
- Feature Engineering (bei zufälligen Operationen)

### 2. Daten-Versionierung

- **Rohdaten:** Unveränderlich in `data/raw/`
- **Verarbeitete Daten:** Werden bei jedem Training neu generiert und gespeichert
- **Empfehlung:** Verwenden Sie DVC (Data Version Control) für Produktionsumgebungen

```bash
# DVC initialisieren (optional)
dvc init
dvc add data/raw/*.parquet
git add data/raw/*.parquet.dvc .dvc/
```

### 3. Environment Management

**Exakte Dependency-Versionen:**
```bash
pip freeze > requirements.txt
```

**Reproduzierbares Environment:**
```bash
pip install -r requirements.txt
```

### 4. MLflow Tracking

Alle Trainings-Runs werden mit MLflow getrackt:
- **Parameter:** Alle Model-Hyperparameter
- **Metriken:** RMSE, MAE, R² auf allen Datasets
- **Artefakte:** Modelle, Feature Importance, Metadata
- **Code Version:** Git commit hash (wenn in Git Repository)

### 5. Konfigurationsdatei

Alle Einstellungen in `config/config.yaml`:
- Model-Parameter
- Feature Engineering Einstellungen
- Daten-Pfade
- API-Konfiguration

### 6. Reproduzierbarer Workflow

```bash
# 1. Environment setup
python -m venv myenv
source myenv/Scripts/activate  # oder myenv\Scripts\activate (Windows)
pip install -r requirements.txt

# 2. Daten vorbereiten
# Kopiere Parquet-Dateien nach data/raw/

# 3. Training
python train.py

# 4. API starten
python api.py
```

### Reproduzierbarkeits-Checklist

- [x] Feste Random Seeds (`random_state=42`)
- [x] Versions-kontrollierte Dependencies (`requirements.txt`)
- [x] Konfigurationsdatei (`config/config.yaml`)
- [x] MLflow Tracking aller Experimente
- [x] Gespeicherte verarbeitete Daten
- [x] Dokumentierte Feature Engineering Pipeline
- [x] Deterministische Daten-Splits

---

## 🔢 Modellversionierung

### Versionierung mit MLflow

Alle Modelle werden automatisch versioniert:

1. **Timestamp-basiert:**
   - Format: `model_{type}_{YYYYMMDD_HHMMSS}.joblib`
   - Beispiel: `model_random_forest_20241126_103000.joblib`

2. **Latest-Link:**
   - Immer verfügbar: `model_latest.joblib`
   - Zeigt auf zuletzt trainiertes Modell

3. **MLflow Registry:**
   - Alle Runs in `mlruns/` gespeichert
   - Browseable über MLflow UI

### Modelle verwalten

#### Verfügbare Modelle anzeigen

```bash
# Via API
curl http://localhost:8000/models

# Via MLflow UI
mlflow ui
# Öffne http://localhost:5000
```

#### Spezifisches Modell verwenden

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"driver_id": "D001", "web_order_id": "ORDER_1"},
    params={"model_version": "random_forest_20241126_103000"}
)
```

#### Modell-Metadaten lesen

```python
import yaml

with open("models/metadata_20241126_103000.yaml") as f:
    metadata = yaml.safe_load(f)
    print(metadata)
```

**Metadata enthält:**
- Model Type
- Training Timestamp
- Metriken (RMSE, MAE, R²)
- Feature Names
- Model Path

### A/B Testing

Laden Sie mehrere Modellversionen und vergleichen Sie Predictions:

```python
# Via API
response_v1 = requests.post(
    "http://localhost:8000/predict",
    json={"driver_id": "D001", "web_order_id": "ORDER_1"},
    params={"model_version": "latest"}
)

response_v2 = requests.post(
    "http://localhost:8000/predict",
    json={"driver_id": "D001", "web_order_id": "ORDER_1"},
    params={"model_version": "random_forest_20241125_120000"}
)

print(f"V1: {response_v1.json()['predicted_service_time']}")
print(f"V2: {response_v2.json()['predicted_service_time']}")
```

---

## 🧪 Testing

```bash
# Alle Tests ausführen
pytest tests/

# Mit Coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📊 Monitoring

### Production Monitoring (Empfehlungen)

1. **Model Performance:**
   - Sammeln Sie Ground Truth Labels
   - Vergleichen Sie mit Predictions aus Logs
   - Berechnen Sie periodisch Metriken

2. **Data Drift:**
   - Überwachen Sie Feature-Verteilungen
   - Alert bei signifikanten Abweichungen

3. **API Performance:**
   - Response Times
   - Error Rates
   - Throughput

---

## 🤝 Contributing

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Erstelle einen Pull Request

---

## 📄 Lizenz

Dieses Projekt wurde für die Flaschenpost SE Bewerbung erstellt.

---

## 👤 Autor

**Ihr Name**
- GitHub: [thefranckt]
- Email: derfrancko@gmail.com

---

## 🙏 Acknowledgments

- Flaschenpost SE für die Aufgabenstellung
- scikit-learn, MLflow, FastAPI Communities

---

## 📞 Support

Bei Fragen oder Problemen:
1. Öffne ein Issue auf GitHub
2. Kontaktiere über Email

---

**Viel Erfolg! 🚀**
