#!/usr/bin/env python3
"""
Debug script to test JSON handling in the AutoML app
"""

import requests
import json
import time

def test_endpoints():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing AutoML endpoints...")
    print("=" * 50)
    
    # Test 1: Basic connectivity
    try:
        response = requests.get(f"{base_url}/test")
        print(f"✅ GET /test: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ GET /test failed: {e}")
    
    # Test 2: Upload endpoint (without file)
    try:
        response = requests.post(f"{base_url}/upload")
        print(f"✅ POST /upload (no file): {response.status_code}")
        if response.status_code == 400:
            print(f"   Expected error: {response.json()}")
    except Exception as e:
        print(f"❌ POST /upload failed: {e}")
    
    # Test 3: Train endpoint with JSON
    try:
        train_data = {"target_column": "test_column"}
        response = requests.post(
            f"{base_url}/train",
            json=train_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"✅ POST /train: {response.status_code}")
        print(f"   Sent data: {train_data}")
        if response.status_code == 400:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ POST /train failed: {e}")
    
    # Test 4: Predict endpoint with JSON
    try:
        predict_data = {"input_data": {"feature1": "value1"}}
        response = requests.post(
            f"{base_url}/predict",
            json=predict_data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"✅ POST /predict: {response.status_code}")
        print(f"   Sent data: {predict_data}")
        if response.status_code == 400:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ POST /predict failed: {e}")
    
    print("=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    print("Waiting for Flask app to start...")
    time.sleep(3)
    test_endpoints()
