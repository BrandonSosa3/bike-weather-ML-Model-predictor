import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Import our feature preparation function
# That gives us X_train, X_test, y_train, y_test, scaler.
from prepare_ml_features import prepare_features_for_ml

def build_and_evaluate_baseline():
    """Build our first linear regression model"""
    
    print("🤖 BUILDING BASELINE LINEAR REGRESSION MODEL")
    print("=" * 50)
    
    # Get our prepared features
    # X_train → Features (inputs) for training. A DataFrame with all the chosen features (temp, humidity, etc.), scaled. (80%)
    # X_test → Features for testing, scaled in the same way, these are same as X_train except this is where we run the model 
    # on unseen data (not in X_train) so we make sure the model does not cheat (20%)
    # y_train → The target values (bike counts) that go with X_train. (the correct answers for X-train basically)
    # y_test = the correct answers for the testing inputs (X-test) (kept hidden from the model until evaluation).
    # scaler → A StandardScaler object that remembers the “recipe” (mean & std) used to scale the training data.

    """ It’s basically a quiz analogy:
        X_train = the questions the student (model) practices on
        y_train = the answer key for practice
        X_test = the final exam questions (new ones!)
        y_test = the final exam answer key (used only for grading, not for studying) """
    X_train, X_test, y_train, y_test, scaler = prepare_features_for_ml()
    
    print(f"🎯 Training linear regression model...")
    
    # LinearRegression() creates linear regression model, at this point the model knows nothing and is an empty container
    # .fit() trains it: finds the best line (or hyperplane, since multiple features) that predicts bike counts.
    """X_train = the features (like temperature, humidity, hour). y_train = the actual answers (bike counts)."""
    # The model tries to find the best line (or plane, or hyperplane) that connects the features to the answers.
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✅ Model trained successfully!")
    
    # Make predictions
    # Here we are asking the trained model, “What bike counts would you predict for the training data (X_train)?” → gives y_train_pred.
    print(f"\n📊 Making predictions...")
    y_train_pred = model.predict(X_train)
    # “What bike counts would you predict for the new unseen test data (X_test)?” → gives y_test_pred.
    y_test_pred = model.predict(X_test)
    
    # Evaluate model performance
    print(f"\n📈 MODEL PERFORMANCE:")
    print("=" * 30)
    
    # Training metrics
    # RMSE (Root Mean Squared Error), Measures how far predictions are from actual values, on average.
    # Penalizes big mistakes more heavily because of the square.
    # Smaller RMSE = better.
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    # MAE (Mean Absolute Error)
    # Average of the absolute differences between predicted and actual.
    # More intuitive: “On average, the model’s prediction is off by this many bikes.”
    # Doesn’t punish big errors as strongly (linear instead of squared).
    train_mae = mean_absolute_error(y_train, y_train_pred)
    # R² measures how much better your model is compared to a dumb baseline (just predicting the average every time).
    # If R² = 1 → perfect predictions (no error).
    # If R² = 0 → model is no better than guessing the average.
    # If R² < 0 → model is worse than guessing the average!
    # so if R2 is 0.915 : “My model explains 91.5% of the variation in bike counts compared to just guessing the average.”
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"📚 Training Set:")
    print(f"  - RMSE: {train_rmse:.1f} bikes")
    print(f"  - MAE:  {train_mae:.1f} bikes")
    print(f"  - R²:   {train_r2:.3f}")
    
    # Test metrics (this is what really matters!)
    # We already trained our model on training data, now we test the model on X_test (the data the model has not seen)
    # remember, y_test_pred are the predictions for that data, and y_test are the real bike counts.
    # So basically here we are using these 3 metrics to compare the real data (y_test) with our predictions (y_test_pred)
    # this line: Finds the average squared difference → that’s Mean Squared Error (MSE). Then takes the square root → that gives Root Mean Squared Error (RMSE).
    # Plain words: “On average, how far off are my predictions, with bigger mistakes punished more?”
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    # Looks at the absolute difference between real and predicted values (ignores +/−, takes abs value). Then averages them
    # Plain words: “On average, my model is wrong by this many bikes.”
    test_mae = mean_absolute_error(y_test, y_test_pred)
    # Calculates R² (coefficient of determination).
    # Measures how much of the variation in bike counts the model explains.
    # If R² = 0.85 → “Model explains 85% of the ups and downs in bike usage.”
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"🧪 Test Set (New Data):")
    print(f"  - RMSE: {test_rmse:.1f} bikes")
    print(f"  - MAE:  {test_mae:.1f} bikes")
    print(f"  - R²:   {test_r2:.3f}")
    
    # Feature importance analysis
    print(f"\n🔍 MOST IMPORTANT FEATURES:")
    print("=" * 30)
    
    # Get feature names and their coefficients
    # pd.DataFrame makes a table with two columns (feature, coefficient)
    # X_train.columns gets the names of all the input features
    # model.coef_ gives the weights/coefficients the regression model learned for each feature
    # positive coeff -> increases predictions, "this feature makes more bikes likely"
    # negative coeff -> decreases predictions, "this feature makes fewer bikes more likely"
    # .sort_values(...) sorts by abs value of the coefficients
    # WHY? - because both strong positive and strong negative features are "influential"
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("Top 10 most influential features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        direction = "↗️" if row['coefficient'] > 0 else "↘️"
        print(f"  {i:2d}. {row['feature']:15s} {direction} {row['coefficient']:8.1f}")
    
    # Create prediction quality visualization
    print(f"\n📊 Creating prediction accuracy visualization...")
    
    # just create a blank canvas for out plots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Actual vs Predicted
    # creates a grid of plots, 1 row, 2 cols
    plt.subplot(1, 2, 1)
    # x-axis is actual bike counts
    # y-axis is model's predicted counts
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue')
    # this draws a dashed red line from the smallest to the largest value 
    # this line represents perfect predictions where predicted = actual
    # if the dots are close to this line the model is predicting well
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Bike Count')
    plt.ylabel('Predicted Bike Count')
    plt.title(f'Predictions vs Reality\nR² = {test_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuals (errors)
    # Residuals = actual - predicted
    # this tells use, what the model predicted (x-axis), how far off the prediction was to the true value (y-axis)
    # above y=0 model predicted bike count too low, below model predicted too high
    plt.subplot(1, 2, 2)
    residuals = y_test - y_test_pred
    # x-axis -> predicted bike counts from our model
    # y-axis -> residuals (errors), each dot equals one prediction
    # if model was perfect every dot would sit on the y=0 line
    plt.scatter(y_test_pred, residuals, alpha=0.5, color='red')
    # this create our horizontal like at 0, this is our perfect prediction reference
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Bike Count')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Errors\n(Should be random around 0)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📁 Performance charts saved as: baseline_model_performance.png")
    
    return model, X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test, scaler = build_and_evaluate_baseline()