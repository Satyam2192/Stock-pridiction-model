import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file, parse_dates=['日付け'])

    # Create features: moving averages, percent change
    df['MA_5'] = df['終値'].rolling(window=5).mean()
    df['MA_20'] = df['終値'].rolling(window=20).mean()
    df['Pct_change'] = df['終値'].pct_change()

    # Drop NaN values
    df.dropna(inplace=True)

    # Scaling
    scaler = MinMaxScaler()
    df[['終値', 'MA_5', 'MA_20', 'Pct_change', '出来高']] = scaler.fit_transform(
        df[['終値', 'MA_5', 'MA_20', 'Pct_change', '出来高']]
    )

    # Save preprocessed data
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('../data/stock_price.csv', '../data/preprocessed_stock_data.csv')
