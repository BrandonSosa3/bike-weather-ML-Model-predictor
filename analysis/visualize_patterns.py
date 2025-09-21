import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

def create_pattern_visualizations():
    """Create visual representations of our discovered patterns"""
    
    # gets us access to all rows in the processed_bike_data table
    engine = create_engine('sqlite:///database/bike_sharing.db')
    df = pd.read_sql_query("SELECT * FROM processed_bike_data", engine)
    
    print("📊 Creating pattern visualizations...")
    
    # Set up the plotting style, makes our plots look nice and simple
    plt.style.use('default')
    sns.set_palette("husl")
    
    # creates a 2 by 2 grid of plots, so 4 subplots total 
    # sets the size of each plot figsize = ...
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # adds the big title across all four plots
    fig.suptitle('Weather & Activity Patterns', fontsize=16, fontweight='bold')
    
    # 1. Weather Comfort vs Usage
    # here we are just creating categories based on our weather_comfort rating we calculated earlier
    # since this vale is between 0 and 1 we are creating each row with a label that is going to be a meaningful word, 
    # instead of a 0.0, we are going to say Awful
    comfort_bins = pd.cut(df['weather_comfort'], bins=5, labels=['Awful', 'Poor', 'OK', 'Good', 'Perfect'])
    # here we calculates the average number of bikes rented per hour for each bin.
    comfort_usage = df.groupby(comfort_bins)['cnt'].mean()
    
    # makes a bar chart for the first plot
    # x-axis is comfort_usage.index which is ['Awful', 'Poor', 'OK', 'Good', 'Perfect']
    # y-axis os comfort_usage.values which is ['cnt'].mean() the average bikes used per comfort
    axes[0, 0].bar(comfort_usage.index, comfort_usage.values, color='skyblue', edgecolor='navy')
    # gives axis labels, and rotates x-axis labels so they do not overlap
    axes[0, 0].set_title('Weather Comfort vs Bike Usage')
    axes[0, 0].set_ylabel('Average Bikes per Hour')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Hourly Usage Patterns
    # groups our rows by hr of the day (0-23), for each hr we are looking for the mean count of bikes used
    hourly_usage = df.groupby('hr')['cnt'].mean()
    # this makes a line chart in the top right axes[0,1]
    # x-axis is hours of the day 
    # y-axis is average bikes per hour
    # marker='o' puts dots on each point, linewidth= 2 makes line thicker
    axes[0, 1].plot(hourly_usage.index, hourly_usage.values, marker='o', linewidth=2, markersize=4)
    axes[0, 1].set_title('Average Usage by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Average Bikes per Hour')
    # sets transparency of gridline
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Weather Situation Impact
    weather_names = {1: 'Clear', 2: 'Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'}
    # we group the rows by the weathersit col (all the 1s together, all the 2s together, ...). Think of sorting rows into four piles by weather code.
    # then from each pile we look at only the cnt col (number of bikes) and take the average for each pile
    weather_usage = df.groupby('weathersit')['cnt'].mean()
    # this loops through each i in weather_usage.index (which is 1,2,3,4) and looksup weather_names[i] from the dict we had earlier
    # to find the word assocaited with each number, the result is weather_labels = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
    weather_labels = [weather_names[i] for i in weather_usage.index]
    
    # create vertical bar graph, the categories for the x-axis-> weather_labels (words)
    # y is the heights of the bars which is going to be the weather_usage.values 
    # (the numeric averages). .values returns a plain NumPy array [200.4, 150.0, 80.1, 25.0].
    # color gives each bar its own color
    axes[1, 0].bar(weather_labels, weather_usage.values, color=['gold', 'lightblue', 'lightcoral', 'darkred'])
    axes[1, 0].set_title('Usage by Weather Condition')
    axes[1, 0].set_ylabel('Average Bikes per Hour')
    # rotates Simple: rotates the x-axis labels (the weather names) by 45 degrees.
    # Why: long labels can overlap; rotation makes them readable.
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    
    # 4. Temperature vs Usage Scatter
    # creates a scatterplot with x-axis as temp and y-axis as bike count
    # alpha=0.1 makes dots transparent so dense areas look darker, s=1 makes dots very tiny
    axes[1, 1].scatter(df['temp'], df['cnt'], alpha=0.1, s=1)
    axes[1, 1].set_title('Temperature vs Usage (Raw Data)')
    axes[1, 1].set_xlabel('Temperature (Normalized)')
    axes[1, 1].set_ylabel('Bike Count')

    # adjusts spacing so plots dont overlap
    plt.tight_layout()
    # popus up our window that displays the charts
    plt.show()
    
    # Print some key insights
    print("\n🔍 KEY INSIGHTS:")
    print(f"• Best weather comfort score yields {comfort_usage['Perfect']:.0f} bikes/hour")
    print(f"• Worst weather comfort score yields {comfort_usage['Awful']:.0f} bikes/hour")
    print(f"• Peak usage hour: {hourly_usage.idxmax()}:00 with {hourly_usage.max():.0f} bikes")
    print(f"• Weather multiplier: Perfect weather = {comfort_usage['Perfect']/comfort_usage['Awful']:.1f}x awful weather")

if __name__ == "__main__":
    create_pattern_visualizations()