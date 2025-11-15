# Python Base Image
FROM python:3.9-slim

# Setze Arbeitsverzeichnis
WORKDIR /app

# Kopiere Requirements
COPY requirements.txt .

# Installiere Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere Projekt-Dateien
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY setup.py .

# Installiere Projekt
RUN pip install -e .

# Erstelle notwendige Verzeichnisse
RUN mkdir -p data/raw data/processed logs mlruns

# Exponiere Port für API
EXPOSE 8000

# Standard Command
CMD ["python", "-m", "src.models.train_model"]
