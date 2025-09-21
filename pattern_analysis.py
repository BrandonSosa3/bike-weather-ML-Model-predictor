import pandas as pd
from sqlalchemy import create_engine

def analyze_weather_activity_patterns():
    """Discover how weather actually affects bike usage"""
    
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("📊 WEATHER & ACTIVITY PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load our enhanced data, the query means to get all rows and cols from processed_bike_data table
    df = pd.read_sql_query("SELECT * FROM processed_bike_data", engine)
    
    print(f"📈 Analyzing {len(df)} hours of bike usage data")
    
    # 1. Weather Impact Analysis
    print("\n🌤️  Weather Impact on Bike Usage:")
    
    # Group by weather situation and see average usage
    # df.groupby("weathersit") means group all of our rows based on the col weathersit, 
    # ['cnt'] means to focus only on the bike counrs column based on weathersit,
    # .agg([...]) means we are asking for 3 things about cnt,
    # 1. mean → average number of bikes used in that weather.
    # 2. std → standard deviation (spread/variability).
    # 3. count → how many rows (hours) of data for that weather.
    # .round(1) rounds the values to one decimal
    weather_impact = df.groupby('weathersit')['cnt'].agg(['mean', 'std', 'count']).round(1)
    # right now the indexing would be in numbers based on the weather like 1 = sunny/hot, 2 = rainy/cloudy, etc.
    # this line just associates those numbers with meaningful names
    weather_names = {1: 'Clear/Sunny', 2: 'Cloudy/Misty', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    # this updates those index numbers with our weather_names
    weather_impact.index = weather_impact.index.map(weather_names)
    
    print(weather_impact)
    
    # 2. Temperature Sweet Spot Analysis
    print("\n🌡️  Temperature Sweet Spot:")
    
    # Create temperature bins and see usage patterns
    # here we are trying to see how bike usage count depends on weather
    # df['temp_bin'] creates a new column in the df called temp_bin that stores which category each rows temp falls into
    # in our data temp is normalized between 0(coldest) and 1(hotest)
    # pd.cut(df['temp'], bins = 5, labels=[...]) will basically divide our 0-1 values into 5 different even parts. 
    # this will be likes bin 1: 0.0->0.2, bin 2: 0.2->0.4.. etc but instead of having these decimals we rename the bins 
    # with labels where bin 1: "very cold"... etc
    df['temp_bin'] = pd.cut(df['temp'], bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
    # df.groupby('temp_bin') groups all those rows by the new temperature cataegories we created
    # ['cnt'] looks only at the bike count col
    # .mean().round(1) calculates the average bike usage for each weather category rounded to one decimal 
    temp_usage = df.groupby('temp_bin')['cnt'].mean().round(1)
    print("Average bike usage by temperature category:")
    # remember temp_usage is a pandas series (not a dict or array).
    # a series is kind of like a one column table with an index ("very cold", "cold".. etc)
    # and a value (the numbes here 65.1, 123.1.. etc). .items() lets you interate through (index, value) pairs in a pandas
    for category, value in temp_usage.items():
        print(f"  {category}: {value} bikes/hour")
    
    # 3. Rush Hour vs Weather Interaction
    print("\n⏰ Rush Hour vs Weather Patterns:")
    # in the df we group the rows by the is_rush_hour and weathersit
    # then we only want the ['cnt'].mean().round(1) mean bike counts for these cols
    rush_weather = df.groupby(['is_rush_hour', 'weathersit'])['cnt'].mean().round(1)
    print("Average usage by rush hour (0=normal, 1=rush) and weather:")
    print(rush_weather)
    
    # 4. Weekend vs Weekday Weather Sensitivity
    print("\n📅 Weekend vs Weekday Weather Sensitivity:")
    # same as above, but here is the understanding of the output: 
    ''' We have 2 weekend categories × 4 weather types = 8 combinations:
        is_weekend=0 (WEEKDAYS):
        weathersit=1 (Clear/Sunny): 206.6 bikes/hour
        weathersit=2 (Cloudy/Misty): 181.6 bikes/hour  
        weathersit=3 (Light Rain): 112.4 bikes/hour
        weathersit=4 (Heavy Rain): 100.0 bikes/hour

        is_weekend=1 (WEEKENDS):
        weathersit=1 (Clear/Sunny): 201.0 bikes/hour
        weathersit=2 (Cloudy/Misty): 157.5 bikes/hour
        weathersit=3 (Light Rain): 109.2 bikes/hour
        weathersit=4 (Heavy Rain): 23.0 bikes/hour'''
    weekend_weather = df.groupby(['is_weekend', 'weathersit'])['cnt'].mean().round(1)
    print("Average usage by weekend (0=weekday, 1=weekend) and weather:")
    print(weekend_weather)
    
    # 5. Our Weather Comfort Index Performance
    print("\n🎯 Weather Comfort Index Validation:")
    
    # See if our comfort index correlates with actual usage
    # rememeber in our pipeline earlier we created a "custom comfort index" that combined temp, humidity, and wind
    # we called that 'weather_comfort'. this is also a continuous score between 0(bad) and 1(better)
    # then we split them into 5 intervals based on the values and label them with words
    comfort_bins = pd.cut(df['weather_comfort'], bins=5, labels=['Awful', 'Poor', 'OK', 'Good', 'Perfect'])
    # then again we group our rows by comfort level and get the mean bike count
    comfort_usage = df.groupby(comfort_bins)['cnt'].mean().round(1)
    
    print("Average bike usage by our weather comfort score:")
    print(comfort_usage)
    
    return df

if __name__ == "__main__":
    patterns_df = analyze_weather_activity_patterns()