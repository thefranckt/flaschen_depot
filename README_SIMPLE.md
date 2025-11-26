# Flaschenpost Service-Zeit-Vorhersage

## Überblick

Machine Learning System zur Vorhersage von Lieferzeiten für Flaschenpost-Bestellungen.

## Schnellstart

### Installation

```bash
# Repository klonen
git clone https://github.com/thefranckt/flaschen_depot.git
cd flaschen_depot

# Virtuelle Umgebung erstellen
python -m venv myenv
source myenv/Scripts/activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Modell trainieren

```bash
python train.py
```

### API starten

```bash
python api.py
```

API verfügbar unter: http://localhost:8000

## API Endpunkte

- `GET /health` - Systemstatus
- `POST /predict` - Vorhersage für einzelne Bestellung
- `GET /metrics` - Modell-Metriken

## Projektstruktur

```
flaschendepot/
├── config/          # Konfiguration
├── data/            # Daten
├── models/          # Trainierte Modelle
├── src/             # Quellcode
├── notebooks/       # Jupyter Notebooks
├── api.py           # REST API
└── train.py         # Modelltraining
```

## Modell-Performance

- **RMSE**: 3.48 Minuten
- **MAE**: 2.67 Minuten
- **R²**: 0.32

## Technologie-Stack

- Python 3.13
- LightGBM
- FastAPI
- MLflow
- Pandas
