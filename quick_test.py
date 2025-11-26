"""
Quick Test - Exécutez ceci pendant que l'API tourne dans un autre terminal
"""
import requests

BASE_URL = "http://localhost:8000"

print("\n=== TEST API ===\n")

# Test Health
try:
    r = requests.get(f"{BASE_URL}/health", timeout=2)
    print(f"✓ Health Check: {r.status_code}")
    print(f"  Models loaded: {r.json()['models_loaded']}")
except Exception as e:
    print(f"✗ API n'est pas accessible: {e}")
    print("\nAssurez-vous que l'API tourne dans un autre terminal:")
    print("  python api.py")
    exit(1)

# Test Prediction
try:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"driver_id": "D001", "web_order_id": "O000001"}
    )
    if r.status_code == 200:
        print(f"\n✓ Prediction: {r.json()['predicted_service_time']:.2f} minutes")
    else:
        print(f"\n✗ Prediction failed: {r.status_code}")
except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n✓ Test terminé!")
print("Visitez http://localhost:8000/docs pour plus de tests interactifs")
