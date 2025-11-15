# Flaschendepot - Data Science MLOps Projekt

## 📋 Projektübersicht

Dieses Projekt implementiert ein vollständiges **MLOps-System** für die Analyse und Vorhersage von Flaschenpfand-Rückgaben. Es demonstriert Best Practices für:

- ✅ Datenverarbeitung und Feature Engineering
- ✅ Machine Learning Model Training
- ✅ Model Evaluation und Vergleich
- ✅ MLflow Experiment Tracking
- ✅ CI/CD mit GitHub Actions
- ✅ DVC für Datenversionierung
- ✅ Docker Containerization
- ✅ FastAPI REST API
- ✅ Umfassende Tests mit pytest

---

## 🏗️ Projekt-Architektur

```
flaschendepot/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # CI/CD Pipeline
├── configs/
│   └── config.yaml            # Projekt-Konfiguration
├── data/
│   ├── raw/                   # Rohdaten
│   ├── processed/             # Verarbeitete Daten
│   └── external/              # Externe Daten
├── docs/                      # Dokumentation
├── models/                    # Trainierte Modelle
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── scripts/
│   ├── train_pipeline.py      # Komplette Training Pipeline
│   └── setup_dvc.sh           # DVC Setup
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI Application
│   ├── data/
│   │   └── make_dataset.py   # Datenverarbeitung
│   ├── features/
│   │   └── build_features.py # Feature Engineering
│   ├── models/
│   │   ├── train_model.py    # Model Training
│   │   └── predict.py        # Vorhersagen
│   └── utils/                # Hilfsfunktionen
├── tests/                     # Unit Tests
├── .dvc/                      # DVC Konfiguration
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Repository klonen

```bash
git clone <repository-url>
cd flaschendepot
```

### 2. Virtual Environment erstellen

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 4. Komplette Pipeline ausführen

```bash
python scripts/train_pipeline.py
```

Dies führt automatisch aus:
- ✅ Datengenerierung (falls nicht vorhanden)
- ✅ Datenverarbeitung und Cleaning
- ✅ Feature Engineering
- ✅ Model Training (mehrere Algorithmen)
- ✅ Model Evaluation
- ✅ Model Speicherung

---

## 📊 Verwendung

### Datenverarbeitung

```bash
python src/data/make_dataset.py
```

### Model Training

```bash
python src/models/train_model.py
```

### Vorhersagen

```python
from src.models.predict import BottlePredictor

predictor = BottlePredictor()

bottle_data = {
    'bottle_type': 'Bier',
    'material': 'Glas',
    'size_category': 'Mittel',
    'volume_ml': 500,
    'deposit_amount': 0.08,
    'weight_grams': 450
}

result = predictor.predict_single(bottle_data)
print(result)
```

### FastAPI Server starten

```bash
uvicorn src.api.main:app --reload
```

API Dokumentation: http://localhost:8000/docs

---

## 🐳 Docker

### Mit Docker Compose

```bash
# Alle Services starten
docker-compose up -d

# Training ausführen
docker-compose up ml-training

# MLflow Server
docker-compose up mlflow-server

# API Server
docker-compose up api
```

### Einzelnes Docker Image

```bash
# Build
docker build -t flaschendepot:latest .

# Run
docker run --rm flaschendepot:latest
```

---

## 📈 MLflow Tracking

```bash
# MLflow UI starten
mlflow ui

# Öffne Browser
http://localhost:5000
```

MLflow tracked automatisch:
- Model Parameters
- Metriken (Accuracy, Precision, Recall, F1)
- Modelle
- Artifacts

---

## 🧪 Testing

### Alle Tests ausführen

```bash
pytest
```

### Mit Coverage Report

```bash
pytest --cov=src --cov-report=html
```

### Spezifische Tests

```bash
# Nur Data Processing Tests
pytest tests/test_data_processing.py

# Nur Model Tests
pytest tests/test_models.py
```

---

## 📦 DVC - Data Version Control

### DVC Setup

```bash
# Initialisiere DVC
dvc init

# Füge Daten hinzu
dvc add data/raw/bottles.csv
dvc add models/model.pkl

# Commit DVC Files
git add data/raw/bottles.csv.dvc models/model.pkl.dvc
git commit -m "Add data to DVC"

# Push zu DVC Remote
dvc push
```

### Daten abrufen

```bash
dvc pull
```

---

## 🔄 CI/CD Pipeline

Die GitHub Actions Pipeline führt automatisch aus:

1. **Testing** (Python 3.9, 3.10, 3.11)
   - Unit Tests
   - Coverage Reports
   
2. **Linting**
   - black (Code Formatting)
   - flake8 (Linting)
   - isort (Import Sorting)
   
3. **Build**
   - Package Building
   - Artifact Upload
   
4. **Docker**
   - Docker Image Build
   - Image Testing

---

## 🎯 Features

### Kategorische Features
- `bottle_type`: Art der Flasche (Bier, Wasser, Saft, etc.)
- `material`: Material (Glas, Plastik, Aluminium)
- `size_category`: Größenkategorie (Klein, Mittel, Groß)

### Numerische Features
- `volume_ml`: Volumen in Millilitern
- `deposit_amount`: Pfandbetrag in Euro
- `weight_grams`: Gewicht in Gramm

### Abgeleitete Features
- `deposit_per_ml`: Pfand pro Milliliter
- `weight_per_ml`: Gewicht pro Milliliter (Dichte)
- `volume_category`: Volumenkategorie
- `material_type_combo`: Material-Typ Kombination

### Zielvariable
- `return_status`: Wurde die Flasche zurückgegeben? (0/1)

---

## 🤖 Unterstützte ML-Algorithmen

- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM (Support Vector Machine)
- XGBoost (via requirements)
- LightGBM (via requirements)
- CatBoost (via requirements)

---

## 📚 Projektstruktur - Detailliert

### `src/data/`
Datenverarbeitungs-Module
- `make_dataset.py`: Laden, Cleaning, Splitting

### `src/features/`
Feature Engineering
- `build_features.py`: Feature-Erstellung, Preprocessing

### `src/models/`
Machine Learning Modelle
- `train_model.py`: Training, Evaluation, Hyperparameter-Tuning
- `predict.py`: Vorhersagen, Batch-Processing

### `src/api/`
FastAPI REST API
- `main.py`: API Endpoints für Vorhersagen

### `tests/`
Unit Tests
- `test_data_processing.py`: Datenverarbeitungs-Tests
- `test_models.py`: Model-Tests

---

## 🔧 Konfiguration

Alle Konfigurationen sind in `configs/config.yaml`:

```yaml
project:
  name: flaschendepot
  version: 0.1.0

data:
  test_size: 0.2
  random_state: 42

model:
  algorithm: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
```

Environment Variables in `.env`:

```env
MLFLOW_TRACKING_URI=./mlruns
MODEL_PATH=./models
DATA_PATH=./data
```

---

## 📊 Metriken

Das Projekt evaluiert Modelle mit:

- **Accuracy**: Gesamtgenauigkeit
- **Precision**: Präzision (gewichtet)
- **Recall**: Recall (gewichtet)
- **F1-Score**: F1-Score (gewichtet)
- **ROC-AUC**: Area Under Curve
- **Confusion Matrix**: Fehlermatrix
- **Classification Report**: Detaillierter Report

---

## 🌐 API Endpoints

Nach dem Start der API (`uvicorn src.api.main:app --reload`):

### Health Check
```http
GET /health
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "bottle_type": "Bier",
  "material": "Glas",
  "size_category": "Mittel",
  "volume_ml": 500,
  "deposit_amount": 0.08,
  "weight_grams": 450
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

[
  {...},
  {...}
]
```

### Model Info
```http
GET /model/info
```

API Dokumentation: http://localhost:8000/docs

---

## 🎓 Best Practices

Dieses Projekt demonstriert:

1. ✅ **Modularität**: Klare Trennung von Concerns
2. ✅ **Reproduzierbarkeit**: Seeds, Versionierung
3. ✅ **Testing**: Umfassende Unit Tests
4. ✅ **Logging**: Strukturiertes Logging
5. ✅ **Dokumentation**: Code-Kommentare, Docstrings
6. ✅ **CI/CD**: Automatisierte Pipeline
7. ✅ **Containerization**: Docker Support
8. ✅ **API**: REST API für Deployment
9. ✅ **Experiment Tracking**: MLflow
10. ✅ **Data Versioning**: DVC

---

## 🤝 Contributing

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

---

## 📝 Lizenz

MIT License - siehe LICENSE Datei

---

## 👤 Autor

**Franz**

---

## 🙏 Danksagungen

- MLflow für Experiment Tracking
- DVC für Data Versioning
- FastAPI für API Framework
- scikit-learn für ML Algorithmen

---

## 📞 Support

Bei Fragen oder Problemen, erstelle bitte ein Issue im Repository.

---

**Happy Machine Learning! 🚀**
