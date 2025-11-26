"""
Test API predictions
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("\n=== Health Check ===")
    print(json.dumps(response.json(), indent=2))
    assert response.status_code == 200


def test_single_prediction():
    """Test single prediction."""
    payload = {
        "driver_id": "D001",
        "web_order_id": "ORDER_12345"  # Replace with actual order ID from your data
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        params={"model_version": "latest"}
    )
    
    print("\n=== Single Prediction ===")
    print(f"Request: {payload}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        assert "predicted_service_time" in response.json()
    else:
        print(f"Error: {response.status_code}")


def test_batch_prediction():
    """Test batch prediction."""
    payload = {
        "model_version": "latest",
        "requests": [
            {"driver_id": "D001", "web_order_id": "ORDER_1"},
            {"driver_id": "D002", "web_order_id": "ORDER_2"},
            {"driver_id": "D003", "web_order_id": "ORDER_3"}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=payload
    )
    
    print("\n=== Batch Prediction ===")
    print(f"Request: {len(payload['requests'])} orders")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['total_count']} predictions")
        print(json.dumps(result, indent=2))
        assert result['total_count'] == len(payload['requests'])
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_get_logs():
    """Test log retrieval."""
    print("\n=== Feature Logs ===")
    response = requests.get(f"{BASE_URL}/logs/features?limit=5")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    print("\n=== Prediction Logs ===")
    response = requests.get(f"{BASE_URL}/logs/predictions?limit=5")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    
    print("\n=== Statistics ===")
    response = requests.get(f"{BASE_URL}/logs/statistics")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("=" * 80)
    print("API TEST SUITE")
    print("=" * 80)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API is running: python api.py")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_get_logs()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Make sure:")
        print("1. API is running (python api.py)")
        print("2. Model is trained (python train.py)")
        print("3. Data is in data/raw/")
