import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def convert_to_float(value):
    """Convert string values like '220.24M' or '-2.56%' to float."""
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        elif value.endswith('B'):
            return float(value[:-1]) * 1_000_000_000
        elif value.endswith('%'):
            return float(value[:-1]) / 100
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    return value

def preprocess_data(input_file, output_file_prefix):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    print("Original columns:", df.columns.tolist())
    
    # Convert date column
    df['日付け'] = pd.to_datetime(df['日付け'])
    df = df.sort_values('日付け')
    
    # Clean and convert numerical columns
    numerical_columns = ['終値', '始値', '高値', '安値', '出来高']
    for col in numerical_columns:
        df[col] = df[col].apply(convert_to_float)
    
    # Convert change rate column (handles the space in the column name)
    change_rate_col = '変化率 %'
    if change_rate_col in df.columns:
        df[change_rate_col] = df[change_rate_col].apply(convert_to_float)
    
    # Create features
    df['MA_5'] = df['終値'].rolling(window=5).mean()
    df['MA_20'] = df['終値'].rolling(window=20).mean()
    df['Volatility'] = df['終値'].rolling(window=5).std()
    
    # Create time-based features
    df['DayOfWeek'] = df['日付け'].dt.dayofweek
    df['Month'] = df['日付け'].dt.month
    
    # Calculate price range
    df['PriceRange'] = df['高値'] - df['安値']
    
    # Drop NaN values
    df.dropna(inplace=True)

    # Prepare features for scaling
    feature_columns = ['終値', '始値', '高値', '安値', '出来高', 'MA_5', 'MA_20', 
                       'Volatility', 'PriceRange']
    
    # Add change rate to features
    if change_rate_col in df.columns:
        feature_columns.append(change_rate_col)
    
    # Scaling
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_prefix), exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, f'{output_file_prefix}_scaler.joblib')

    # Prepare final feature set
    X = df[feature_columns + ['DayOfWeek', 'Month']]
    y = df['終値']
    dates = df['日付け']

    # Split the data while preserving time order
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    # Save preprocessed data
    X_train.to_csv(f'{output_file_prefix}_X_train.csv', index=False)
    X_test.to_csv(f'{output_file_prefix}_X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{output_file_prefix}_y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{output_file_prefix}_y_test.csv', index=False)
    pd.DataFrame(dates_train).to_csv(f'{output_file_prefix}_dates_train.csv', index=False)
    pd.DataFrame(dates_test).to_csv(f'{output_file_prefix}_dates_test.csv', index=False)
    
    print(f"\nData preprocessing completed successfully!")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Features used: {feature_columns + ['DayOfWeek', 'Month']}")

if __name__ == "__main__":
    input_file = '../data/stock_price.csv'
    output_prefix = '../data/preprocessed_stock_data'
    
    try:
        preprocess_data(input_file, output_prefix)
    except Exception as e:
        print(f"An error occurred: {str(e)}")