import os
from dotenv import load_dotenv
from pathlib import Path

def download_bike_sharing_data():
    """Download the bike sharing dataset from Kaggle"""
    
    # Load our environment variables FIRST
    load_dotenv()
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
    
    # NOW import kaggle after setting credentials
    import kaggle
    
    # Make sure our raw data folder exists
    raw_data_path = Path('data/raw')
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading bike sharing dataset...")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            'lakshmi25npathi/bike-sharing-dataset',
            path='data/raw/',
            unzip=True
        )
        print("✅ Dataset downloaded successfully!")
        
        # List what we got
        print("\n📁 Files downloaded:")
        for file in raw_data_path.glob('*'):
            print(f"  - {file.name}")
            
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    download_bike_sharing_data()