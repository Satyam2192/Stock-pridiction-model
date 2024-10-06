import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_model():
    # Load the preprocessed data
    X_test = pd.read_csv('../data/preprocessed_stock_data_X_test.csv')
    y_test = pd.read_csv('../data/preprocessed_stock_data_y_test.csv')
    dates_test = pd.read_csv('../data/preprocessed_stock_data_dates_test.csv')
    
    # Convert dates back to datetime
    dates_test['日付け'] = pd.to_datetime(dates_test['日付け'])

    # Load the saved model
    model = load_model('../models/final_stock_price_model.keras')

    # Reshape X_test
    X_test_reshaped = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Make predictions
    predictions = model.predict(X_test_reshaped)

    # Load the scaler and inverse transform the predictions and actual values
    scaler = joblib.load('../data/preprocessed_stock_data_scaler.joblib')
    
    # Create dummy arrays for inverse transform
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy[:, 0] = predictions.flatten()  # Assuming '終値' is the first column
    predictions_original = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_test.values.flatten()
    y_test_original = scaler.inverse_transform(dummy)[:, 0]

    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Model Accuracy: {r2*100:.2f}%")

    # Ensure results directory exists
    os.makedirs('../results', exist_ok=True)

    # Save evaluation results
    evaluation_results = pd.DataFrame({
        'Date': dates_test['日付け'],
        'Actual_Prices': y_test_original,
        'Predicted_Prices': predictions_original,
        'Error': y_test_original - predictions_original
    })
    evaluation_results.to_csv('../results/evaluation_results.csv', index=False)

    # Plot predictions vs actuals
    plt.figure(figsize=(15, 7))
    plt.plot(dates_test['日付け'], y_test_original, label='Actual Prices', color='blue')
    plt.plot(dates_test['日付け'], predictions_original, label='Predicted Prices', color='orange')
    plt.title('Stock Price Predictions vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/predictions_vs_actuals.png')
    plt.close()

if __name__ == "__main__":
    try:
        evaluate_model()
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")

