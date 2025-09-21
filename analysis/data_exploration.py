import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def explore_data_quality():
    """Investigate data quality issues in our dataset"""
    
    # Connect to database
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("DATA QUALITY INVESTIGATION")
    print("-" * 50)
    
    # Load all data for exploration
    df = pd.read_sql_query("SELECT * FROM raw_bike_data", engine)
    
    print(f"Dataset Overview:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Check for missing values
    print(f"\nMissing Values:")
    # df.isnull returns a df of True/False values (true if the cell is null/NaN)
    # .sum on that -> counts how many Trues(missing values) are in each column
    missing = df.isnull().sum()
    # loops through each item in the missing count
    for col, count in missing.items():
        # this checks if any are missing then we print the column name, number of missing values, and the percentage of missing entries in that column
        if count > 0:
            print(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")
    # checks if there are no missing values across all columns
    if missing.sum() == 0:
        print(" No missing values found!")
    
    # df.duplicated() checks each row in your df and tells you whether its a duplciate of a previous row
    # will return a Pandas Series and if the row is a duplicate with have true if it is a duplicate of previous row, 
    # FALSE IF IT UNIQUE SO FAR
    # .sum(), in python true = 1, false = 0, so in this case it adds all true values (adds all duplicate rows)
    duplicates = df.duplicated().sum()
    # prints number of duplicate rows
    print(f"\nDuplicate Records: {duplicates}")
    
    # Examine data ranges and potential outliers
    print(f"\nData Ranges (Weather Variables):")
    # define the list of column names for the weather
    weather_cols = ['temp', 'atemp', 'hum', 'windspeed']
    # For each, prints the minimum and maximum values to check for unrealistic or outlier values.
    for col in weather_cols:
        print(f"  - {col}: {df[col].min():.3f} to {df[col].max():.3f}")
    
    print(f"\nActivity Ranges:")
    # defines the column names for bike usage and activity
    activity_cols = ['casual', 'registered', 'cnt']
    # for each column print the min and max and average values. 
    for col in activity_cols:
        print(f"  - {col}: {df[col].min()} to {df[col].max()} (avg: {df[col].mean():.1f})")
    # return original df so it can be used in the next step of pipeline
    return df

if __name__ == "__main__":
    df = explore_data_quality()