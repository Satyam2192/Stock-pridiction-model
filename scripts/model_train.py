import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def load_data(X_file, y_file):
    X = pd.read_csv(X_file).values
    y = pd.read_csv(y_file).values.flatten()
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model():
    # Load data
    X_train, y_train = load_data(
        '../data/preprocessed_stock_data_X_train.csv',
        '../data/preprocessed_stock_data_y_train.csv'
    )
    X_test, y_test = load_data(
        '../data/preprocessed_stock_data_X_test.csv',
        '../data/preprocessed_stock_data_y_test.csv'
    )

    # Create model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    # Prepare callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        '../models/best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Save the final model
    model.save('../models/final_stock_price_model.keras')
    
    return history

if __name__ == "__main__":
    try:
        os.makedirs('../models', exist_ok=True)
        history = train_model()
        print("Model training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")