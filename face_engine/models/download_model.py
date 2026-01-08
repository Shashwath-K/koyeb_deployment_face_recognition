# download_model.py - Download MobileFaceNet pretrained weights
import torch
import gdown
import os

def download_mobilefacenet():
    """Download MobileFaceNet pretrained weights"""
    os.makedirs('models', exist_ok=True)
    model_path = 'models/mobilefacenet_model.pth'
    
    # MobileFaceNet pretrained weights URL
    url = 'https://drive.google.com/uc?id=1MduR0DcK4GqZzYqg7L9c-M8T3zZZjQVr'
    
    print("Downloading MobileFaceNet weights...")
    gdown.download(url, model_path, quiet=False)
    
    # Verify download
    if os.path.exists(model_path):
        print(f"✓ Model downloaded to {model_path}")
        
        # Test loading
        model = torch.load(model_path, map_location='cpu')
        print(f"✓ Model loaded successfully")
        print(f"  Model keys: {list(model.keys()) if isinstance(model, dict) else 'state_dict'}")
    else:
        print("✗ Download failed")

if __name__ == '__main__':
    download_mobilefacenet()