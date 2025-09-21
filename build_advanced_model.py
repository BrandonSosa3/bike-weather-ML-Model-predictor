import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Import our baseline function and feature preparation
from build_baseline_model import build_and_evaluate_baseline
from prepare_ml_features import prepare_features_for_ml

def build_random_forest_model():
    """Build a Random Forest model to beat our baseline - best for non linear data"""
    # Overfitting = it memorizes the training data, so it works well on old data but poorly on new data.
    # How to spot overfitting = training error: very low, test error: much higher, big gap between training and test performance 
    """What is random forest model? Start with a single tree, lets say you want to predict how many bikes will be rented today.
        First you ask: is it raining? yes or no. then is it rush hour? yes or no. The tree keeps splitting the data based on yes/no answers
        until it makes a prediction. Now, this is cool but trees are limited and subject to overfitting. A random forest builds lots 
        of trees. (100,200,etc). Each tree sees a slightly different subset of data and sometimes only a subset of features. each tree
        will make its own prediction. For regression (like bike counts) random forest averages all the tree predictions. For classification
        (yes/no) categories it uses a majority vote."""
    
    print("🌲 BUILDING RANDOM FOREST MODEL")
    print("=" * 50)
    
    # Get prepared features
    X_train, X_test, y_train, y_test, scaler = prepare_features_for_ml()
    
    # Build Random Forest model object 
    print("🌳 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,      # Number of trees: uses 100 decision trees
        random_state=42,       # For reproducible results, just a random seed. makes sure when we run code we get same result for comparisons
        max_depth=15,          # Prevent overfitting, limits the depth so it cannot memorize the data
        # This means: a split in the tree (a decision like “temperature > 70°F?”) only happens if there are at least 10 data points > 70 F.
        min_samples_split=10,
        n_jobs=-1              # Use all CPU cores, makes training faster
    )
    
    # fits the random forest to the training data
    rf_model.fit(X_train, y_train)
    print("✅ Random Forest trained!")
    
    # Make predictions
    # uses the trained model to predict bike counts on both training and test sets
    # .predict(...) when you train a model with .fit, the model learns patterns in the training data
    # (relationships between features X_train and the target y_train). .fit then takes a set of feature inputs X and uses learned patterns
    # predict target values. 
    rf_train_pred = rf_model.predict(X_train)  # predictions for data the model saw during training (based on what you learned, whatis prediction?)
    rf_test_pred = rf_model.predict(X_test)    # predictions for the new/unseen data (predict counts for totally new daya you did not see in training)
    
    # Calculate performance metrics
    # so after training we now have predictions from the model, rf_train_pred, rf_test_pred
    # we now check how close are these predictions to y_train, y_test
    # We are using RMSE and R^2 in this case to check this

    # here we do RMSE on the real bike counts (y_train) and the predicted ones (rf_train_pred)
    # this is basically just (prediction-actual)^2 for each data point. then we take the sqrt to get back to the correct units
    # remember this metric RMSE tells us "On average, the model's predictions are off by about ___ bikes"
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    # here we do R^2 metric on the real bikes counts vs the predicted ones. 
    # R² measures how much of the variation in bike counts is explained by the model.
    # this is basically just 1 - (error from our model/error if we just guessed average every time)
    rf_train_r2 = r2_score(y_train, rf_train_pred)

    # here we do the same for the test metrics. This shows how well the model generalizes to unseen data
    # this is the true measure of usefullness
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_test_r2 = r2_score(y_test, rf_test_pred)

    """Why both metrics matter
        RMSE → tells us the average size of the error (practical, in bike units).
        R² → tells us the relative quality of the model compared to a dumb guess.
        Together they give a full picture:
        RMSE answers "How wrong are we, on average, in bikes?"
        R² answers "How much of the bike demand can we explain with features?"""
    
    print(f"\n🌲 RANDOM FOREST PERFORMANCE:")
    print("=" * 35)
    print(f"📚 Training Set:")
    print(f"  - RMSE: {rf_train_rmse:.1f} bikes")
    print(f"  - R²:   {rf_train_r2:.3f}")
    
    print(f"🧪 Test Set:")
    print(f"  - RMSE: {rf_test_rmse:.1f} bikes")
    print(f"  - R²:   {rf_test_r2:.3f}")
    
    # Compare with baseline
    print(f"\n⚡ COMPARISON WITH LINEAR REGRESSION:")
    print("=" * 40)
    
    # Get baseline performance (we know these from previous run)
    baseline_test_r2 = 0.367
    baseline_test_rmse = 175.2
    
    # here we just simply find the difference between the linear regression model and the random forest model for each metric
    r2_improvement = rf_test_r2 - baseline_test_r2
    rmse_improvement = baseline_test_rmse - rf_test_rmse
    
    print(f"🎯 R² Score:")
    print(f"  Linear Regression: {baseline_test_r2:.3f}")
    print(f"  Random Forest:     {rf_test_r2:.3f}")
    print(f"  Improvement:       {r2_improvement:+.3f}")
    
    print(f"🎯 RMSE (Lower is Better):")
    print(f"  Linear Regression: {baseline_test_rmse:.1f} bikes")
    print(f"  Random Forest:     {rf_test_rmse:.1f} bikes")
    print(f"  Improvement:       {rmse_improvement:+.1f} bikes")
    
    # Feature importance from Random Forest
    print(f"\n🌟 RANDOM FOREST FEATURE IMPORTANCE:")
    print("=" * 40)
    
    # Unlike linear regression which has coefficients, Random Forests measure feature importance
    # A feature’s importance = how much it helps reduce prediction error across all the decision trees.
    # Higher value → that feature splits the data in useful ways.
    # So here we make a DataFrame with 2 cols, one for "feature", one for "importance"
    # then we sort by importance (biggest first)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    


    print("Top 10 most important features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:15s} {row['importance']:6.3f}")
    
    # here we return some useful things like the model, R^2, RMSE
    return rf_model, rf_test_r2, rf_test_rmse

if __name__ == "__main__":
    rf_model, rf_r2, rf_rmse = build_random_forest_model()