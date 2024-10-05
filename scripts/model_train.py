import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_data(input_file):
    df = pd.read_csv(input_file)
    X = df[['終値', 'MA_5', 'MA_20', 'Pct_change', '出来高']].values
    y = df['終値'].values
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y):
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, batch_size=1, epochs=10)
    return model

if __name__ == "__main__":
    X, y = load_data('../data/preprocessed_stock_data.csv')
    model = train_model(X, y)
    model.save('stock_price_model.h5')
