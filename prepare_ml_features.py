import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler

def prepare_features_for_ml():
    """Prepare our features for machine learning models"""
    """It loads cleaned bike-sharing data, picks which columns to use as inputs and output for a machine learning model, 
    converts text-like columns into numbers the model can use, splits the data into a training set and a test set (time-based split), 
    scales the numeric features so they’re on the same scale, and returns everything ready to feed into an ML algorithm."""
    
    engine = create_engine('sqlite:///database/bike_sharing.db')
    
    print("🤖 PREPARING FEATURES FOR MACHINE LEARNING")
    print("=" * 50)
    
    # Load our processed data
    df = pd.read_sql_query("SELECT * FROM processed_bike_data", engine)
    
    print(f"📊 Starting with {len(df)} rows and {len(df.columns)} columns")
    
    # Select features for our model
    print("\n🎯 Selecting features for prediction...")
    
    # We select some features that should help predict bike usage best
    feature_columns = [
        # Weather features
        'temp', 'atemp', 'hum', 'windspeed', 'weathersit',
        # Time features  
        'hr', 'weekday', 'mnth', 'season', 'holiday', 'workingday',
        # Our engineered features
        'weather_comfort', 'is_rush_hour', 'day_of_year', 'is_weekend'
    ]
    
    # Target variable (the column of what we want to predict)
    target = 'cnt'  # Total bike count
    
    # Create feature matrix (X) and target vector (y)
    # .copy() makes a separate copy so we do not accidentally change the original df later on
    # X is the feature matrix is a table containing only the columns from the feature_columns
    X = df[feature_columns].copy()
    # y is the target vector which is a single column with the cnt values
    y = df[target].copy()
    
    # Prints confirmation and then lists each chosen feature, numbered.
    print(f"✅ Selected {len(feature_columns)} features:")
    for i, feature in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {feature}")
    
    # Handle categorical variables
    print(f"\n🏷️  Encoding categorical variables...")
    
    # One-hot encode weather situation (1,2,3,4 → separate binary columns)
    # convert non numerical columns in numerics the ML model can use
    # completes one hot encoding (see explanation on chat)
    # pd.get_dummies performs this one hot encoding on the weathersit column
    # basically we cannot just give the columns direct values 1-4 because the model might then 4 is greater than 1
    # or something like that. so we use one hot instead
    weather_dummies = pd.get_dummies(X['weathersit'], prefix='weather')
    # X.drop('weathersit', axis=1) removes the original weathersit column from X.
    # pd.concat([...], axis=1) attaches (horizontally) the new weather_* dummy columns to the rest of X.
    # After this, X contains the original numeric features plus the new binary weather columns.
    X = pd.concat([X.drop('weathersit', axis=1), weather_dummies], axis=1)
    
    print(f"✅ Created weather dummy variables: {list(weather_dummies.columns)}")
    
    # Split data for training and testing
    print(f"\n📊 Splitting data for training and testing...")
    
    # Time-aware split: use first 80% for training, last 20% for testing
    # This simulates real-world scenario where we predict future from past
    """You want to predict bike rentals tomorrow.
        If you shuffle and mix the data, your training set might accidentally contain “tomorrow”’s data already.]
        That’s cheating — the model has seen the future!
        With time-aware split, you only train on the past and test on the future, just like real-world forecasting
        Think of it like studying for an exam:
        You study chapters 1–8 (training).
        Then the exam asks you questions from chapters 9–10 (testing).
        If you mixed them randomly, you’d be training on some chapter 10 content — but that’s unfair since in real life you can’t peek into the future."""
    # dteday is a col of dates but right now it is just strings like "2011-01-01"
    # pd.to_datetime turns those strings into actual date objects that Python can understand and compare (like 2011-01-01 as a real date, not just text).
    df['dteday'] = pd.to_datetime(df['dteday'])
    # .quantile(0.8) means: find the date that’s 80% of the way through the timeline.
    # Example: if the dataset covers Jan 2011 → Dec 2012, the 80% point might be around September 2012.
    # We’ll use this as the cut-off point: before this = training, after this = testing.
    split_date = df['dteday'].quantile(0.8)
    
    # a mask is just a col of true false values
    # train_mask = true for the rows where the date is before or on the split date
    train_mask = df['dteday'] <= split_date
    # test_mask is true for rows where the date is after the split date
    test_mask = df['dteday'] > split_date

    # Here we apply the masks to the split data
    # Remember X is the feature cols 
    # Remember y is the target (the bike count cnt)
    # X_train → feature rows before split date
    X_train = X[train_mask]
    # X_test → feature rows after split date
    X_test = X[test_mask]
    # y_train → target values before split date
    y_train = y[train_mask]
    # y_test → target values after split date
    y_test = y[test_mask]
    
    print(f"📈 Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"📉 Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"🗓️  Split date: {split_date}")
    
    # Scale features for better model performance
    print(f"\n⚖️  Scaling features...")


    # ML models work better when all numbers are on a similar scale
    # Example: temp might go from 0–40, but humidity goes from 0–100.
    # The model might think humidity is "more important" just because the numbers are bigger.
    # StandardScaler fixes this: It makes each feature have mean = 0 and standard deviation = 1. So all features are roughly on the same scale.
    scaler = StandardScaler()

    # Fit → look at the training data and calculate the average (mean) and spread (standard deviation) for each column (feature).
    # Transform → use those numbers to scale the training data so every column has: mean = 0, SD = 1
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with proper column names
    # After scaling, the data comes back as a NumPy array (just numbers, no column names).
    # This line wraps it back into a DataFrame so we:
    # Keep column names (temp, humidity, windspeed, …).
    # Keep row indexes (so each row still lines up with the right date).
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"✅ Features scaled to mean=0, std=1")
    
    # Show summary statistics
    print(f"\n📊 Target variable summary:")
    print(f"  Training set - Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
    print(f"  Test set - Mean: {y_test.mean():.1f}, Std: {y_test.std():.1f}")
    
    print(f"\n🎯 Feature matrix shape: {X_train_scaled.shape}")
    print(f"📋 Final feature list:")
    for i, col in enumerate(X_train_scaled.columns, 1):
        print(f"  {i:2d}. {col}")
    

    # here we return all important pieces so they can be used the ML model
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_features_for_ml()