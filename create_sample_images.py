import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_images():
    """Create sample images for the dataset"""
    
    # Create images directory if it doesn't exist
    os.makedirs('sample_data/images', exist_ok=True)
    
    # Sample images data
    images_data = [
        {"filename": "cat1.jpg", "label": "cat", "color": (255, 165, 0), "text": "Cat 1"},
        {"filename": "dog1.jpg", "label": "dog", "color": (255, 215, 0), "text": "Dog 1"},
        {"filename": "cat2.jpg", "label": "cat", "color": (0, 0, 0), "text": "Cat 2"},
        {"filename": "dog2.jpg", "label": "dog", "color": (139, 69, 19), "text": "Dog 2"},
        {"filename": "cat3.jpg", "label": "cat", "color": (128, 128, 128), "text": "Cat 3"},
        {"filename": "dog3.jpg", "label": "dog", "color": (0, 0, 0), "text": "Dog 3"},
        {"filename": "cat4.jpg", "label": "cat", "color": (255, 255, 255), "text": "Cat 4"},
        {"filename": "dog4.jpg", "label": "dog", "color": (255, 0, 0), "text": "Dog 4"},
        {"filename": "cat5.jpg", "label": "cat", "color": (0, 128, 0), "text": "Cat 5"},
        {"filename": "dog5.jpg", "label": "dog", "color": (0, 0, 255), "text": "Dog 5"},
    ]
    
    for img_data in images_data:
        # Create a 224x224 image (standard size for many models)
        img = Image.new('RGB', (224, 224), img_data["color"])
        draw = ImageDraw.Draw(img)
        
        # Add some geometric shapes to make images more interesting
        if img_data["label"] == "cat":
            # Draw cat-like shapes (circles for head, triangles for ears)
            draw.ellipse([50, 50, 150, 150], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.polygon([(100, 30), (80, 60), (120, 60)], fill=(255, 255, 255), outline=(0, 0, 0))
            draw.polygon([(100, 30), (120, 60), (140, 60)], fill=(255, 255, 255), outline=(0, 0, 0))
        else:
            # Draw dog-like shapes (rectangles for body, circles for head)
            draw.rectangle([60, 80, 140, 160], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            draw.ellipse([80, 40, 120, 80], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Add text label
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        draw.text((10, 10), img_data["text"], fill=(0, 0, 0), font=font)
        
        # Save the image
        img_path = f'sample_data/images/{img_data["filename"]}'
        img.save(img_path)
        print(f"Created {img_path}")
    
    print("Sample images created successfully!")

if __name__ == "__main__":
    create_sample_images()
