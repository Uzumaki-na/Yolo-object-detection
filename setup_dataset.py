import os
import kaggle
from zipfile import ZipFile
import shutil

def setup_dataset():
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    
    # Download dataset using Kaggle API
    print("Downloading Pascal VOC 2012 dataset...")
    kaggle.api.dataset_download_files('huanghanchina/pascal-voc-2012', 
                                    path='data', 
                                    unzip=True)
    
    # Move files to correct locations if needed
    voc_path = "data/VOC2012"
    if not os.path.exists(voc_path):
        os.makedirs(voc_path, exist_ok=True)
    
    print("Dataset downloaded and extracted successfully!")
    print(f"Dataset location: {os.path.abspath(voc_path)}")

if __name__ == "__main__":
    # First, set up Kaggle credentials
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("Please enter your Kaggle credentials:")
        api_token = input("Enter your Kaggle API token: ")
        
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
            f.write(api_token)
        
        # Set permissions
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 600)
    
    setup_dataset()