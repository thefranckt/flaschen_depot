"""
Script de test simple pour l'API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("\n=== TEST API SERVICE TIME PREDICTION ===\n")

# Test 1: Health Check
print("1. Health Check...")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    if r.status_code == 200:
        print("   ✓ API est en bonne santé")
        data = r.json()
        print(f"   Modèles chargés: {data['models_loaded']}")
        print(f"   Versions disponibles: {data['available_versions']}")
    else:
        print(f"   ✗ Erreur: {r.status_code}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")
    print("   Assurez-vous que l'API tourne: python api.py")
    exit(1)

# Test 2: Prediction
print("\n2. Test de prédiction...")
try:
    payload = {
        "driver_id": "D001",
        "web_order_id": "O000001"
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    if r.status_code == 200:
        data = r.json()
        print("   ✓ Prédiction réussie")
        print(f"   Temps de service prédit: {data['predicted_service_time']:.2f} minutes")
        print(f"   Driver ID: {data['driver_id']}")
        print(f"   Order ID: {data['web_order_id']}")
        print(f"   Version du modèle: {data['model_version']}")
    else:
        print(f"   ✗ Erreur: {r.status_code}")
        print(f"   Réponse: {r.text}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")

# Test 3: Metrics
print("\n3. Métriques du modèle...")
try:
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    if r.status_code == 200:
        data = r.json()
        print("   ✓ Métriques récupérées")
        print(f"   RMSE: {data.get('rmse', 'N/A')}")
        print(f"   MAE: {data.get('mae', 'N/A')}")
        print(f"   R²: {data.get('r2', 'N/A')}")
    else:
        print(f"   ✗ Erreur: {r.status_code}")
except Exception as e:
    print(f"   ✗ Erreur: {e}")

print("\n=== TESTS TERMINÉS ===\n")
