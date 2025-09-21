import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from prepare_ml_features import prepare_features_for_ml
from datetime import datetime

def create_prediction_interface():
    """Create an interface to test predictions with different scenarios"""
    
    print("🔮 BIKE USAGE PREDICTION INTERFACE")
    print("=" * 50)
    
    # Load and train our Random Forest model
    print("🌲 Loading Random Forest model...")
    X_train, X_test, y_train, y_test, scaler = prepare_features_for_ml()
    

    # This is the same Random Forest we constructed in the build_advanced_model.py 
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1
    )
    

    # Again we did this same thing in the model file
    # This is where the training happens, the random forest looks at all the input features in X_train
    # It then tries to map them to the correct outputs in y_train (the actual bike counts). Each decision tree
    # learns patterns (like “If it’s 8 AM and sunny, predict high bike usage”). After all trees are trained,
    # the model is ready to make predictions on NEW data
    rf_model.fit(X_train, y_train)
    print("✅ Model loaded and ready for predictions!")
    

    def predict_bike_usage(hour, temp, humidity, windspeed, weather_situation, 
                          is_weekend=False, is_holiday=False):
        """
        Predict bike usage for given conditions, this is where we are going to use our trained model to make predictions
        You call this function whenever you want to predict how many bikes will be used under certain conditions.
        
        # These are the input parameter we are going to be using
        “Hey, given these weather and time conditions, how many bikes do you think people will rent?”

        Parameters:
        - hour: 0-23 (hour of day)
        - temp: 0-1 (normalized temperature, 0.5 ≈ comfortable)
        - humidity: 0-1 (0=dry, 1=very humid)
        - windspeed: 0-1 (0=calm, 1=very windy)
        - weather_situation: 1=Clear, 2=Cloudy, 3=Light Rain, 4=Heavy Rain
        - is_weekend: True/False
        - is_holiday: True/False
        """
        
        # Create feature vector matching our training data
        # Here we are building a DataFrame (table) with one row of inputs and all of these columns
        # We are doing this because the model expects inputs in the exact same format (columns) as during training. 
        # If we do not match this the model will not work.
        features = pd.DataFrame({
            'temp': [temp],
            'atemp': [temp * 0.9],  # Approximate "feels like" temperature
            'hum': [humidity],
            'windspeed': [windspeed],
            'hr': [hour],
            'weekday': [6 if is_weekend else 2],  # Saturday vs Tuesday
            'mnth': [6],  # Assume June (middle of year)
            'season': [2],  # Spring
            'holiday': [1 if is_holiday else 0],
            'workingday': [0 if (is_weekend or is_holiday) else 1],
            'weather_comfort': [calculate_weather_comfort(temp, humidity, windspeed)],
            'is_rush_hour': [1 if hour in [7,8,9,17,18,19] else 0],
            'day_of_year': [150],  # Middle of year
            'is_weekend': [1 if is_weekend else 0],
            'weather_1': [1 if weather_situation == 1 else 0],
            'weather_2': [1 if weather_situation == 2 else 0],
            'weather_3': [1 if weather_situation == 3 else 0],
            'weather_4': [1 if weather_situation == 4 else 0]
        })
        
        # Scale features using the same scaler from training
        # This is the scaler object used in training as well, so we use it here too
        features_scaled = scaler.transform(features)
        
        # Make prediction
        # This passes the scaled features into the trained rf_model
        # The model looks at all its decision trees, makes predictions, and averages them
        # .predict() returns a list/array. Since we only gave one row, we grab the first result with [0]
        prediction = rf_model.predict(features_scaled)[0]
        
        # Sometimes the model could predict a negative number (like -12 bikes/hour), which makes no sense.
        # Here we just make sure the lowest value that could ever be returned is 0
        return max(0, prediction)  # Ensure non-negative prediction
    
    def calculate_weather_comfort(temp, humidity, windspeed):
        """Calculate weather comfort index"""
        # Here we are creating a single comfort score for the weather, which is based on temp, humidity, and windspeed
        # higher score = more comfortable conditions
        # We normalize between 0-1, and then multiply by two the penalize more the farther away from 0.5 you are
        # Then we subtract by 1 to flip it where 1 = perfect close to 0=bad
        # Ex: temp = 0.5-> 1 - abs(0.5-0.5) * 2 = 1 (perfect comfort)
        temp_comfort = 1 - abs(temp - 0.5) * 2
        # here we combine all there things into one comfort score.
        # so temp * 0.4 means temp accounts for 40% of comfort. humidity is 30% as well as windspeed
        return (temp_comfort * 0.4 + (1 - humidity) * 0.3 + (1 - windspeed) * 0.3)
    
    # Test different scenarios
    print(f"\n🧪 TESTING DIFFERENT SCENARIOS:")
    print("=" * 35)
    

    # Here we are creating a list of dictionaries, each separate dict is one weather/day scenario
    # each scenario has a name, then the inputs for predict_bike_usage method. 
    scenarios = [
        {
            'name': '🌞 Perfect Summer Day (Rush Hour)',
            'hour': 8, 'temp': 0.6, 'humidity': 0.3, 'windspeed': 0.1, 
            'weather_situation': 1, 'is_weekend': False, 'is_holiday': False
        },
        {
            'name': '🌧️ Rainy Morning Commute',
            'hour': 8, 'temp': 0.4, 'humidity': 0.8, 'windspeed': 0.3,
            'weather_situation': 3, 'is_weekend': False, 'is_holiday': False
        },
        {
            'name': '☀️ Nice Weekend Afternoon',
            'hour': 14, 'temp': 0.7, 'humidity': 0.4, 'windspeed': 0.2,
            'weather_situation': 1, 'is_weekend': True, 'is_holiday': False
        },
        {
            'name': '🌨️ Heavy Rain Weekend',
            'hour': 14, 'temp': 0.3, 'humidity': 0.9, 'windspeed': 0.6,
            'weather_situation': 4, 'is_weekend': True, 'is_holiday': False
        },
        {
            'name': '🌅 Early Morning Weekday',
            'hour': 6, 'temp': 0.5, 'humidity': 0.6, 'windspeed': 0.2,
            'weather_situation': 2, 'is_weekend': False, 'is_holiday': False
        }
    ]
    
    # Here we are running predictions for each scenario
    # results = [] starts an empty list to store outputs
    # then we loop through each scenario
    results = []
    for scenario in scenarios:
        # scenario.items() gives all key value pairs in the scenario dictionary
        # {k: v for k, v in scenario.items() if k != 'name'} → removes the "name" field (because predict_bike_usage doesn’t expect it).
        # ** → unpacks the dictionary so each key/value gets passed as a function argument.
        prediction = predict_bike_usage(**{k: v for k, v in scenario.items() if k != 'name'})
        # This saves the scenarios name and predicted bike usage in the results list
        results.append({'scenario': scenario['name'], 'prediction': prediction})
        # Neatly prints the scenario name and prediction rounded to the nearest whole number
        print(f"{scenario['name']:30s} → {prediction:6.0f} bikes/hour")
    
    # Interactive prediction function
    # Returns two things for later use: 1. The predict_bike_usage function itself
    # The results list of test scenarios and predictions 
    return predict_bike_usage, results

def create_hourly_prediction_chart():
    """Create a chart showing predicted usage throughout the day, for each hour of the day """
    
    print(f"\n📊 Creating hourly prediction chart...")
    
    # Get prediction function
    # Calls another function which returns predict_func: a function we can call to get bike usage predictions, _: the scenarios list
    # which we do not need here so it is ignored using _
    predict_func, _ = create_prediction_interface()
    
    # Predict for each hour on a nice weekday
    # Here we create a list of all hours for a day [0-23]
    hours = list(range(24))
    # Then we have two empty lists
    good_weather_predictions = []  # will hold predictions for good weather
    bad_weather_predictions = []   # will hold predictions for bad weather
    
    # Loops through each hour in the , by the end of this loop we have:
    # 1. good_weather_predictions = list of 24 (one for each hours in nice weather)
    # 2. bad_weather_predictions = list of 24 predictions (for rainy weather)
    for hour in hours:
        # Good weather scenario, calls predict_func and stores the prediction in good_weather_predictions
        good_pred = predict_func(hour=hour, temp=0.6, humidity=0.4, windspeed=0.1,
                               weather_situation=1, is_weekend=False, is_holiday=False)
        good_weather_predictions.append(good_pred)
        
        # Bad weather scenario, does the same as above for bad weather
        bad_pred = predict_func(hour=hour, temp=0.4, humidity=0.8, windspeed=0.4,
                              weather_situation=3, is_weekend=False, is_holiday=False)
        bad_weather_predictions.append(bad_pred)
    
    # Create the chart
    # this line creates a blank chart
    plt.figure(figsize=(12, 6))
    #draws two lines, green line for good weather predictions, red line for bad weather predictions
    plt.plot(hours, good_weather_predictions, 'g-', linewidth=3, label='☀️ Clear Weather', marker='o')
    plt.plot(hours, bad_weather_predictions, 'r-', linewidth=3, label='🌧️ Rainy Weather', marker='s')
    
    plt.title('Predicted Bike Usage Throughout the Day\n(Weekday Scenarios)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Bikes per Hour')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    
    # Add rush hour shading, orange shaded rectangles to highlight rush hours
    plt.axvspan(7, 9, alpha=0.2, color='orange', label='Morning Rush')
    plt.axvspan(17, 19, alpha=0.2, color='orange', label='Evening Rush')
    
    plt.tight_layout()
    plt.savefig('hourly_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📁 Hourly prediction chart saved as: hourly_predictions.png")

# This section only runs when you run the file directly (not when its imported).
if __name__ == "__main__":
    predict_func, scenarios = create_prediction_interface()
    create_hourly_prediction_chart()
    
    print(f"\n🎯 Your model is ready! You can now test custom scenarios.")
    print(f"💡 Try calling: predict_func(hour=17, temp=0.7, humidity=0.3, windspeed=0.1, weather_situation=1)")