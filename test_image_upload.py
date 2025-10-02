import requests
import os

def test_image_upload():
    """Test image upload functionality"""
    
    # Test with one of our sample images
    image_path = "sample_data/images/cat1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test the upload endpoint
    url = "http://127.0.0.1:5000/upload"
    
    with open(image_path, 'rb') as f:
        files = {'file': ('cat1.jpg', f, 'image/jpeg')}
        print(f"Sending file: cat1.jpg")
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except:
        print(f"Response text: {response.text}")
    
    # Check if file was saved
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        files = os.listdir(uploads_dir)
        print(f"Files in uploads directory: {files}")
    else:
        print("Uploads directory does not exist")

if __name__ == "__main__":
    test_image_upload()
