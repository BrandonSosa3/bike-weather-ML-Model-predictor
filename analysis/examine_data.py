import pandas as pd
from pathlib import Path

def examine_raw_data():
    """Take a first look at our downloaded data"""
    
    data_path = Path('data/raw')
    
    print("🔍 Examining our raw data files...\n")
    
    # Look at hour.csv (this is our main dataset)
    if (data_path / 'hour.csv').exists():
        print("📊 HOUR.CSV - Our main dataset:")
        
        # Load just the first few rows to peek
        df = pd.read_csv(data_path / 'hour.csv')
        
        print(f"  - Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  - Date range: {df['dteday'].min()} to {df['dteday'].max()}")
        print("\n  - Column names:")
        for i, col in enumerate(df.columns):
            print(f"    {i+1:2d}. {col}")
        
        print("\n  - First 3 rows:")
        # our columns mean the following:
        # 1. instant - just gives us the row number
        # 2. dteday - the actual date we are looking at
        # 3. season - 1=winter, 2=spring, 3=summer, 4=fall
        # 4. yr - 0 =2011, 1=2012
        # 5. mnth - months (1-12)
        # 6. hr - hour of day (0-23)
        # 7. holiday - 1 if holiday, 0 if not
        # 8. weekday - 0=sunday, 1=monday, ..., 6=saturday
        # 9. workingday - 1 if weekday and not holiday, 0 otherwise
        # 10. weathersit: 1=Clear, 2=Cloudy, 3=Light Rain/Snow, 4=Heavy Rain/Snow
        # 11. temp: Normalized temperature (0-1 scale) - 0.24 would be cold weather on this scale
        # 12. atemp: "Feels like" temperature (0-1 scale)
        # 13. hum: Humidity (0-1 scale)
        # 14. windspeed: Wind speed (0-1 scale)
        # 15. casual: Count of casual/unregistered users
        # 16. registered: Count of registered users
        # 17. cnt: Total count (casual + registered)
        print(df.head(3).to_string())
        
    else:
        print("❌ hour.csv not found!")

if __name__ == "__main__":
    examine_raw_data()