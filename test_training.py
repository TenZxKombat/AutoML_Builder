import requests
import json

def test_training():
    """Test model training with image data"""
    
    # Test the training endpoint
    url = "http://127.0.0.1:5000/train"
    
    data = {
        'target_column': 'label'
    }
    
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    try:
        result = response.json()
        print(f"Response: {result}")
    except:
        print(f"Response text: {response.text}")

if __name__ == "__main__":
    test_training()

