import os
from scripts.preprocess import preprocess_data
from scripts.model_train import train_model, load_data

if __name__ == "__main__":
    # Define file paths
    raw_data_path = 'data/stock_price.csv'
    preprocessed_data_path = 'data/preprocessed_stock_data.csv'

    # Preprocess data
    preprocess_data(raw_data_path, preprocessed_data_path)

    # Train model
    X, y = load_data(preprocessed_data_path)
    model = train_model(X, y)
    model.save('stock_price_model.h5')
