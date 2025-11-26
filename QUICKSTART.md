# Quick Start Guide

## Schnellstart in 5 Minuten

### 1. Setup

```bash
# Virtual Environment erstellen und aktivieren
python -m venv myenv
source myenv/Scripts/activate  # Windows Git Bash
# oder: myenv\Scripts\activate  # Windows CMD/PowerShell

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Daten hinzufügen

Kopiere die 4 Parquet-Dateien nach `data/raw/`:
- `orders.parquet`
- `articles.parquet`
- `service_times.parquet`
- `driver_order_mapping.parquet`

### 3. Training

```bash
python train.py
```

Das Training:
- Lädt und bereinigt Daten
- Erstellt Features
- Trainiert Modell (Random Forest)
- Speichert Modell in `models/`
- Loggt alles zu MLflow

**Dauer:** ~2-5 Minuten (je nach Datengröße)

### 4. API starten

```bash
python api.py
```

API läuft auf: http://localhost:8000

### 5. API testen

**Browser:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "driver_id": "D001",
        "web_order_id": "YOUR_ORDER_ID"  # Order ID aus deinen Daten
    }
)
print(response.json())
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict?model_version=latest" \
  -H "Content-Type: application/json" \
  -d '{"driver_id": "D001", "web_order_id": "YOUR_ORDER_ID"}'
```

### 6. Weitere Tools

**MLflow UI:**
```bash
mlflow ui
# Öffne http://localhost:5000
```

**Jupyter Notebook (EDA):**
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## Tipps

### Order IDs finden

```python
import pandas as pd

orders = pd.read_parquet("data/raw/orders.parquet")
print(orders['web_order_id'].head(10))
```

### Modell neu trainieren

```bash
# Mit anderem Modelltyp
# Editiere config/config.yaml: model.type = "xgboost"
python train.py
```

### Logs inspizieren

```bash
# Via API
curl "http://localhost:8000/logs/predictions?limit=10"

# Via SQLite
sqlite3 logs/prediction_store.db "SELECT * FROM prediction_logs LIMIT 10;"
```

### Troubleshooting

**Problem:** Import-Fehler
```bash
# Lösung: Virtual Environment aktivieren
source myenv/Scripts/activate
pip install -r requirements.txt
```

**Problem:** "Model not found"
```bash
# Lösung: Erst trainieren
python train.py
```

**Problem:** "Order not found"
```bash
# Lösung: Verwende Order ID aus deinen Daten
python -c "import pandas as pd; print(pd.read_parquet('data/raw/orders.parquet')['web_order_id'].head())"
```

---

## Projekt-Struktur Übersicht

```
flaschendepot/
├── config/config.yaml          # Konfiguration
├── data/raw/*.parquet          # Rohdaten (hier reinlegen!)
├── notebooks/*.ipynb           # EDA
├── src/*.py                    # Modules
├── train.py                    # Training Script
├── api.py                      # API
├── models/                     # Gespeicherte Modelle
├── logs/                       # Feature & Prediction Logs
└── mlruns/                     # MLflow Tracking
```

---

## Nächste Schritte

1. ✅ Lese die vollständige [README.md](README.md)
2. 🔍 Führe EDA aus: `jupyter notebook`
3. 🚀 Experimentiere mit Hyperparametern in `config/config.yaml`
4. 📊 Analysiere Logs in `logs/*.db`
5. 🎯 Teste verschiedene Modelltypen (Random Forest, XGBoost, LightGBM)

---

**Viel Erfolg! 🚀**
