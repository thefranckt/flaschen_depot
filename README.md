# Flaschen Depot - MLOps Project

Ein umfassendes MLOps-Projekt für die Verwaltung und Vorhersage von Flaschenpfand-Daten.

## 📋 Projektübersicht

Flaschen Depot ist ein vollständiges Machine Learning Operations (MLOps) Projekt, das Best Practices für die Entwicklung, das Training, die Bereitstellung und die Überwachung von Machine Learning-Modellen demonstriert. Das Projekt konzentriert sich auf die Verwaltung eines Flaschendepots mit maschinellem Lernen.

## 🏗️ Projektstruktur

```
flaschen_depot/
├── .github/
│   └── workflows/          # CI/CD Pipeline-Definitionen
│       └── ci.yml          # GitHub Actions Workflow
├── configs/                # Konfigurationsdateien
│   └── config.yaml         # Hauptkonfiguration
├── data/
│   ├── raw/               # Rohdaten
│   ├── processed/         # Verarbeitete Daten
│   └── external/          # Externe Datenquellen
├── logs/                  # Protokolldateien
├── models/                # Trainierte Modelle
├── notebooks/             # Jupyter Notebooks für EDA
│   └── 01_eda.ipynb      # Explorative Datenanalyse
├── scripts/               # Utility-Skripte
│   └── train.py          # Training-Pipeline
├── src/
│   └── flaschen_depot/
│       ├── data/          # Daten-Module
│       │   ├── __init__.py        # Daten-Ingestion
│       │   └── preprocessing.py   # Datenvorverarbeitung
│       ├── models/        # Modell-Module
│       │   └── __init__.py        # Modelltraining und -bewertung
│       ├── utils/         # Utility-Module
│       │   ├── __init__.py        # Logging-Utilities
│       │   └── config.py          # Konfigurationsloader
│       ├── api.py         # FastAPI-Anwendung
│       └── __init__.py    # Paket-Initialisierung
├── tests/                 # Test-Suite
│   ├── conftest.py       # Pytest-Konfiguration
│   ├── test_data.py      # Daten-Tests
│   ├── test_preprocessing.py  # Preprocessing-Tests
│   └── test_models.py    # Modell-Tests
├── .dvc/                  # DVC-Konfiguration
├── .dvcignore            # DVC-Ignorierdatei
├── .gitignore            # Git-Ignorierdatei
├── docker-compose.yml    # Docker Compose-Konfiguration
├── Dockerfile            # Docker-Image-Definition
├── pyproject.toml        # Python-Projekt-Konfiguration
├── requirements.txt      # Python-Abhängigkeiten
├── setup.py              # Paket-Setup
└── README.md             # Projektdokumentation
```

## 🚀 Funktionen

### MLOps-Komponenten

- **Data Management**: Daten-Ingestion, Validierung und Versionierung mit DVC
- **Model Training**: Automatisierte Training-Pipelines mit MLflow-Tracking
- **Model Serving**: REST API mit FastAPI für Modellvorhersagen
- **CI/CD**: Automatisierte Tests und Deployment mit GitHub Actions
- **Containerization**: Docker und Docker Compose für reproduzierbare Umgebungen
- **Monitoring**: Logging und Modellüberwachung
- **Testing**: Umfassende Unit-Tests mit pytest

### Kernfunktionalitäten

- Automatische Datenverarbeitung und Feature Engineering
- Training von Classification- und Regression-Modellen
- REST API für Batch- und Einzelvorhersagen
- Experiment-Tracking mit MLflow
- Datenversioning mit DVC
- Code-Qualitätssicherung (Black, Flake8, isort)

## 📦 Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip
- Docker (optional, für containerisierte Bereitstellung)
- Git

### Lokale Installation

1. Repository klonen:
```bash
git clone https://github.com/thefranckt/flaschen_depot.git
cd flaschen_depot
```

2. Virtuelle Umgebung erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
pip install -e .
```

### Docker-Installation

```bash
# Image erstellen
docker build -t flaschen_depot:latest .

# Container ausführen
docker run -p 8000:8000 flaschen_depot:latest
```

### Mit Docker Compose

```bash
# Alle Services starten (API + MLflow)
docker-compose up -d

# Services stoppen
docker-compose down
```

## 🎯 Verwendung

### 1. Daten vorbereiten

```python
from flaschen_depot.data import DataIngestion

# Daten-Ingestion initialisieren
data_ingestion = DataIngestion('data/raw')

# Beispieldaten erstellen
df = data_ingestion.create_sample_data(n_samples=1000)
data_ingestion.save_data(df, 'bottles.csv')
```

### 2. Modell trainieren

```bash
# Training-Pipeline ausführen
python scripts/train.py
```

Oder programmatisch:

```python
from flaschen_depot.data import DataIngestion
from flaschen_depot.data.preprocessing import DataPreprocessor
from flaschen_depot.models import ModelTrainer

# Daten laden und vorverarbeiten
data_ingestion = DataIngestion()
df = data_ingestion.create_sample_data(1000)

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
df_encoded = preprocessor.encode_categorical(df_clean, ['bottle_type', 'condition'])

# Features vorbereiten
X, y = preprocessor.prepare_features(df_encoded, 'condition')
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

# Modell trainieren
trainer = ModelTrainer()
trainer.train_classifier(X_train, y_train)
trainer.evaluate_classifier(X_test, y_test)
trainer.save_model('model.pkl')
```

### 3. API starten

```bash
# API lokal starten
uvicorn flaschen_depot.api:app --host 0.0.0.0 --port 8000 --reload
```

API-Dokumentation ist verfügbar unter:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Vorhersagen treffen

```bash
# Einzelvorhersage
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bottle_type": 1,
    "volume_ml": 500,
    "deposit_amount": 0.25,
    "condition": 2,
    "return_count": 5,
    "last_return_days": 30
  }'
```

### 5. MLflow UI öffnen

```bash
mlflow ui
```

Öffnen Sie http://localhost:5000 im Browser, um Experimente zu verfolgen.

## 🧪 Tests ausführen

```bash
# Alle Tests ausführen
pytest tests/

# Mit Coverage-Report
pytest tests/ --cov=src/flaschen_depot --cov-report=html

# Spezifische Tests
pytest tests/test_data.py
```

## 🔍 Code-Qualität

```bash
# Code formatieren
black src/flaschen_depot

# Imports sortieren
isort src/flaschen_depot

# Linting
flake8 src/flaschen_depot
```

## 📊 Daten-Versionierung

```bash
# DVC initialisieren (falls noch nicht geschehen)
dvc init

# Daten tracken
dvc add data/raw/bottles.csv

# Änderungen committen
git add data/raw/bottles.csv.dvc .gitignore
git commit -m "Add data tracking"

# Daten pushen
dvc push
```

## 🔧 Konfiguration

Die Hauptkonfiguration befindet sich in `configs/config.yaml`. Sie können folgende Aspekte konfigurieren:

- Modell-Hyperparameter
- Datenpfade
- MLflow-Einstellungen
- API-Konfiguration
- Logging-Level

## 📈 MLOps-Workflow

1. **Data Ingestion**: Daten aus verschiedenen Quellen laden
2. **Data Preprocessing**: Daten bereinigen und Features erstellen
3. **Model Training**: Modell mit MLflow-Tracking trainieren
4. **Model Evaluation**: Modellleistung bewerten
5. **Model Registry**: Modell in MLflow registrieren
6. **Model Serving**: Modell über REST API bereitstellen
7. **Monitoring**: Modellleistung in Produktion überwachen
8. **Retraining**: Automatisches Retraining bei Performanceabfall

## 🤝 Beitragen

Beiträge sind willkommen! Bitte:

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committen Sie Ihre Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Pushen Sie zum Branch (`git push origin feature/AmazingFeature`)
5. Öffnen Sie einen Pull Request

## 📝 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

## 👥 Autoren

- Flaschen Depot Team

## 🙏 Danksagungen

- MLflow für Experiment-Tracking
- DVC für Daten-Versionierung
- FastAPI für API-Framework
- Scikit-learn für ML-Algorithmen

## 📞 Kontakt

Bei Fragen oder Feedback wenden Sie sich bitte an das Projektteam.

---

**Version**: 0.1.0  
**Status**: In Entwicklung  
**Letzte Aktualisierung**: November 2025