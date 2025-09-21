import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from datetime import datetime

def load_csv_to_database():
    """Load our CSV data into the database"""
    
    # Check if files exist
    csv_path = Path('data/raw/hour.csv')
    if not csv_path.exists():
        print("❌ CSV file not found! Run download_data.py first.")
        return
    
    # Connect to database
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("📖 Reading CSV file...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from CSV")
    
    # Convert the date column from string to actual date
    print("🔄 Converting date format...")
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    print("💾 Inserting data into database...")
    
    # Insert data into database uses pandas function to_sql
    df.to_sql(
        'raw_bike_data', # name of sql table
        engine,          # sqlalchemy engine that knows how to talk to db
        if_exists='replace',  # Replace table if it already has data
        index=False,          # Don't add pandas index as a column
        method='multi'        # Faster insertion for large datasets
    )
    
    print("✅ Data loaded successfully!")
    
    # Verify what we loaded
    # COUNT(*) - asks how many total rows are in the table
    # MIN(dteday) - the earliest date 
    # MAX(dteday) - the lastest date
    # AVG(cnt) - the avergae of the cnt column (the number of bike rentals per hour/day)
    print("\n🔍 Verification:")
    verification_query = """
    SELECT 
        COUNT(*) as total_rows,
        MIN(dteday) as earliest_date,
        MAX(dteday) as latest_date,
        AVG(cnt) as avg_bike_count
    FROM raw_bike_data
    """
    # runs the sql query in the db
    # result holds a pandas dataframe
    result = pd.read_sql_query(verification_query, engine)
    print(f"  - Total rows: {result['total_rows'][0]}")
    print(f"  - Date range: {result['earliest_date'][0]} to {result['latest_date'][0]}")
    print(f"  - Average hourly bike rentals: {result['avg_bike_count'][0]:.1f}")

if __name__ == "__main__":
    load_csv_to_database()