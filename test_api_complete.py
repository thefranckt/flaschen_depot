"""
Test complet de l'API avec démarrage automatique
"""
import subprocess
import time
import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_PROCESS = None

def start_api():
    """Démarre l'API en arrière-plan"""
    global API_PROCESS
    print("Démarrage de l'API...")
    API_PROCESS = subprocess.Popen(
        [sys.executable, "api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent
    )
    
    # Attendre que l'API soit prête
    for i in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("✓ API démarrée et prête\n")
                return True
        except:
            time.sleep(1)
    
    print("✗ Timeout: L'API n'a pas démarré\n")
    return False

def stop_api():
    """Arrête l'API"""
    global API_PROCESS
    if API_PROCESS:
        API_PROCESS.terminate()
        API_PROCESS.wait(timeout=5)
        print("\n✓ API arrêtée")

def test_health():
    """Test 1: Health Check"""
    print("1. Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print("   ✓ API en bonne santé")
            data = r.json()
            print(f"   Modèles chargés: {data['models_loaded']}")
            print(f"   Versions disponibles: {data['available_versions']}")
            return True
        else:
            print(f"   ✗ Erreur: {r.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False

def test_prediction():
    """Test 2: Prédiction"""
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
            print(f"   Temps prédit: {data['predicted_service_time']:.2f} minutes")
            print(f"   Driver: {data['driver_id']}")
            print(f"   Order: {data['web_order_id']}")
            print(f"   Version: {data['model_version']}")
            return True
        else:
            print(f"   ✗ Erreur {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False

def test_metrics():
    """Test 3: Métriques du modèle"""
    print("\n3. Métriques du modèle...")
    try:
        r = requests.get(f"{BASE_URL}/metrics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print("   ✓ Métriques récupérées")
            print(f"   Version: {data['model_version']}")
            print(f"   Type: {data['model_type']}")
            print(f"   Timestamp: {data['timestamp']}")
            print(f"\n   Test Metrics:")
            print(f"   - RMSE: {data['test_metrics'].get('rmse', 'N/A'):.2f} min")
            print(f"   - MAE: {data['test_metrics'].get('mae', 'N/A'):.2f} min")
            print(f"   - R²: {data['test_metrics'].get('r2', 'N/A'):.4f}")
            
            fi = data.get('feature_importance', [])
            if fi:
                print(f"\n   Top 3 features:")
                for i, feat in enumerate(fi[:3], 1):
                    print(f"   {i}. {feat['feature']}: {feat['importance']:.0f}")
            return True
        else:
            print(f"   ✗ Erreur {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print(f"   ✗ Erreur: {e}")
        return False

def main():
    print("\n=== TEST COMPLET DE L'API ===\n")
    
    try:
        # Démarrer l'API
        if not start_api():
            return 1
        
        # Exécuter les tests
        results = []
        results.append(("Health Check", test_health()))
        results.append(("Prédiction", test_prediction()))
        results.append(("Métriques", test_metrics()))
        
        # Résumé
        print("\n" + "="*50)
        print("RÉSUMÉ DES TESTS")
        print("="*50)
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status} - {name}")
        
        print(f"\nRésultat: {passed}/{total} tests réussis")
        
        return 0 if passed == total else 1
        
    except KeyboardInterrupt:
        print("\n\n✗ Tests interrompus par l'utilisateur")
        return 1
    finally:
        stop_api()

if __name__ == "__main__":
    sys.exit(main())
