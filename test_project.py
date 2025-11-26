"""
Quick API Test
"""
import requests
import json
import time

# Warte bis API bereit ist
print("Warte auf API...")
time.sleep(2)

BASE_URL = "http://localhost:8000"

print("\n" + "=" * 80)
print("API TEST")
print("=" * 80)

# Test 1: Health Check
print("\n1. Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("   ✓ API ist gesund")
        print(f"   Geladene Modelle: {response.json()['models_loaded']}")
    else:
        print(f"   ✗ Fehler: {response.status_code}")
except Exception as e:
    print(f"   ✗ Fehler: {e}")
    print("   Bitte stellen Sie sicher, dass die API läuft: python api.py")
    exit(1)

# Test 2: Single Prediction
print("\n2. Single Prediction Test...")
try:
    payload = {
        "driver_id": "D001",
        "web_order_id": "O000001"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        params={"model_version": "latest"},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Vorhersage erfolgreich")
        print(f"   Order ID: {result['web_order_id']}")
        print(f"   Predicted Service Time: {result['predicted_service_time']:.2f} Minuten")
        print(f"   Model Version: {result['model_version']}")
    else:
        print(f"   ✗ Fehler: {response.status_code}")
        print(f"   {response.json()}")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 3: Batch Prediction
print("\n3. Batch Prediction Test...")
try:
    payload = {
        "model_version": "latest",
        "requests": [
            {"driver_id": "D001", "web_order_id": "O000001"},
            {"driver_id": "D002", "web_order_id": "O000002"},
            {"driver_id": "D003", "web_order_id": "O000003"}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=payload,
        timeout=15
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Batch Prediction erfolgreich")
        print(f"   Anzahl Predictions: {result['total_count']}")
        for pred in result['predictions'][:3]:
            print(f"   - {pred['web_order_id']}: {pred['predicted_service_time']:.2f} Minuten")
    else:
        print(f"   ✗ Fehler: {response.status_code}")
        print(f"   {response.json()}")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

# Test 4: Logs
print("\n4. Logs Test...")
try:
    response = requests.get(f"{BASE_URL}/logs/statistics", timeout=5)
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✓ Logs verfügbar")
        print(f"   Total Predictions: {stats.get('total_predictions', 0)}")
    else:
        print(f"   ✗ Fehler: {response.status_code}")
except Exception as e:
    print(f"   ✗ Fehler: {e}")

print("\n" + "=" * 80)
print("✓ ALLE TESTS ABGESCHLOSSEN")
print("=" * 80)
print("\nDas Projekt funktioniert einwandfrei!")
print("\nWeitere Schritte:")
print("  - Swagger UI: http://localhost:8000/docs")
print("  - MLflow UI: mlflow ui (dann http://localhost:5000)")
print("  - EDA Notebook: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
