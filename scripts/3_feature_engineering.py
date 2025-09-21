import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def create_enhanced_features():
    """Create meaningful features from our clean data, basically creating new columns where we can use the data from original db and make it more
    intuitive and useful. ex: we combine all weather cols to make just one overall weather comfort score"""
    
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("🔧 FEATURE ENGINEERING")
    print("=" * 50)
    
    # Load data
    df = pd.read_sql_query("SELECT * FROM raw_bike_data", engine)
    
    print(f"📊 Starting with {len(df.columns)} columns")
    
    # Convert dteday to datetime for easier manipulation
    # Converts the dteday column (a string like "2011-01-01") into a proper datetime64 object so we can extract day, month, weekday, etc.
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Create weather comfort features
    print("🌤️  Creating weather comfort features...")
    
    # Temperature comfort (closer to 0.5 = more comfortable), does some tricky math
    df['temp_comfort'] = 1 - abs(df['temp'] - 0.5) * 2
    
    # Wind discomfort (higher wind = less comfortable), basically just uses windspeed as a direct discomfort measure
    df['wind_discomfort'] = df['windspeed']
    
    # Overall weather comfort index, produces a single weather comfort value or index between 0-1 combining all the cols related
    df['weather_comfort'] = (df['temp_comfort'] * 0.4 + 
                            (1 - df['hum']) * 0.3 + 
                            (1 - df['wind_discomfort']) * 0.3)
    
    # Create time-based features
    print("⏰ Creating time-based features...")
    
    # Rush hour indicators, creates a brand new column in the df called is_rush_hour "|" means or here
    # .astype(int) just converts True -> 1 and false -> 0
    """hr   is_rush_hour
        6    0
        7    1
        8    1
        9    1
        17   1
        18   1
        19   1
        20   0"""
    # checks if the hour is a rush hour, 
    df['is_rush_hour'] = ((df['hr'].between(7, 9)) | 
                         (df['hr'].between(17, 19))).astype(int)
    
    # Day of year  between 1-365 (for seasonal trends beyond just season number)
    df['day_of_year'] = df['dteday'].dt.dayofyear
    
    # Weekend indicator, sunday's value in the db is 0 and saturday is 6. if the day of week is 0 or 6 we have true.
    # .astype(int) converts true -> 1 (weekend) and false -> 0 (weekday)
    df['is_weekend'] = (df['weekday'].isin([0, 6])).astype(int)
    
    # now we see how many new columns (features) we added. since we started originally with 17
    print(f"✅ Created {len(df.columns) - 17} new features")
    print(f"📊 Now have {len(df.columns)} total columns")
    
    # Show some examples of our new features
    print(f"\n🔍 Sample of new features:")
    # list of selected columns we want to show
    sample_cols = ['hr', 'temp', 'windspeed', 'hum', 'weather_comfort', 'is_rush_hour', 'cnt']
    # .head() takes the first 5 rows only, .to_string() prints them in a neat aligned table format
    print(df[sample_cols].head().to_string())
    
    # NEW: Save our enhanced features to the database permanently
    print("\n💾 Saving enhanced features to database...")
    
    # Write the enhanced DataFrame to a new table called 'processed_bike_data'
    # if_exists='replace' means if the table already exists, overwrite it completely
    # index=False means don't save the pandas row numbers as a column
    # method='multi' makes the insertion faster for large datasets
    df.to_sql(
        'processed_bike_data',        # table name
        engine,                       # database connection
        if_exists='replace',          # overwrite existing table
        index=False,                  # don't include row numbers
        method='multi'                # faster insertion method
    )
    
    print("✅ Enhanced data saved to 'processed_bike_data' table!")
    
    # Verify what we saved by counting rows in the new table
    verification = pd.read_sql_query(
        "SELECT COUNT(*) as total_rows FROM processed_bike_data", 
        engine
    )
    print(f"📊 Verified: {verification['total_rows'][0]} rows saved to database")
    
    # Quick peek at what's now in our database
    print(f"\n🗃️  Database now contains:")
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(tables_query, engine)
    for table in tables['name']:
        count_query = f"SELECT COUNT(*) as count FROM {table};"
        count = pd.read_sql_query(count_query, engine)['count'][0]
        print(f"  - {table}: {count} rows")
    
    return df

"""If this file is run directly (python features.py), it calls the function and stores the enhanced DataFrame in enhanced_df.
If the file is imported in another script, nothing runs automatically — but you can still call create_enhanced_features() yourself."""
if __name__ == "__main__":
    enhanced_df = create_enhanced_features()