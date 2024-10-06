# Stock Price Prediction Model

## Project Overview
This project implements a deep learning model to predict stock prices using historical time-series data from NTT stock. The model utilizes LSTM (Long Short-Term Memory) neural networks to capture temporal dependencies in the stock price movements.

## Features
- Data preprocessing and feature engineering
- Time series analysis with LSTM neural networks
- Cross-validation using TimeSeriesSplit
- Model evaluation with multiple metrics
- Visualization of predictions vs actual prices

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
statsmodels
```

## Project Structure
```
stock-prediction/
├── data/
│   ├── stock_price.csv
│   └── preprocessed_stock_data_*.csv
├── models/
│   ├── final_stock_price_model.keras
│   └── improved_stock_price_model.keras
├── results/
│   ├── evaluation_results.csv
│   └── predictions_vs_actuals.png
├── scripts/
│   ├── preprocess.py
│   ├── model_train.py
│   ├── evaluate.py
│   └── model_improvement.py
└── requirements.txt
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Satyam2192/Stock-pridiction-model.git)
cd stock-prediction
```

2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv env
env\Scripts\activate
```

For Linux/macOS:
```bash
python3 -m venv env
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess the data:
```bash
python scripts/preprocess.py
```

2. Train the model:
```bash
python scripts/model_train.py
```

3. Evaluate the model:
```bash
python scripts/evaluate.py
```

## Model Performance
- Mean Squared Error (MSE): 2.01
- Root Mean Squared Error (RMSE): 1.42
- Mean Absolute Error (MAE): 1.15
- R-squared (R²): 0.9964
- Model Accuracy: 99.64%

## Data Features
The model uses the following features:
- Closing Price (終値)
- Opening Price (始値)
- High Price (高値)
- Low Price (安値)
- Volume (出来高)
- 5-day Moving Average
- 20-day Moving Average
- Volatility
- Price Range
- Change Rate (変化率 %)
- Day of Week
- Month

## Results Visualization


### 0. Data Set

| 日付け (Date)   | 終値 (Closing Price) | 始値 (Opening Price) | 高値 (High Price) | 安値 (Low Price) | 出来高 (Volume) | 変化率 % (Change Rate %) |
|----------------|----------------------|----------------------|-------------------|------------------|-----------------|-------------------------|
| 2024-08-01     | 156.3                | 159.3                | 159.4             | 156.1            | 79.15M          | -2.56%                  |
| 2024-07-31     | 160.4                | 158.2                | 160.7             | 158.1            | 173.91M         | 1.07%                   |
| 2024-07-30     | 158.7                | 158.8                | 159.2             | 158.0            | 138.14M         | -0.63%                  |
| 2024-07-29     | 159.7                | 158.7                | 160.2             | 158.4            | 126.28M         | 1.14%                   |
| 2024-07-26     | 157.9                | 159.3                | 159.6             | 157.9            | 155.08M         | -0.13%                  |
| 2024-07-25     | 158.1                | 157.0                | 159.0             | 156.8            | 190.62M         | -0.25%                  |
| 2024-07-24     | 158.5                | 160.0                | 160.4             | 158.4            | 168.57M         | -1.37%                  |
| 2024-07-23     | 160.7                | 161.6                | 161.7             | 159.8            | 165.79M         | 0.50%                   |
| 2024-07-22     | 159.9                | 159.5                | 159.9             | 157.9            | 136.12M         | 0.57%                   |

- Preprocessed Data

| 日付け (Date)   | 終値 (Closing Price) | 始値 (Opening Price) | 高値 (High Price) | 安値 (Low Price) | 出来高 (Volume) | 変化率 % (Change Rate %) | MA_5  | MA_20 | Pct_change |
|----------------|----------------------|----------------------|-------------------|------------------|-----------------|-------------------------|-------|-------|------------|
| 2024-07-04     | 0.4565775009160865    | 159.0                | 159.1             | 157.2            | 0.16597673649914219 | -0.57%                  | 0.45814160605610793 | 0.478035136118934 | 0.4660724385636594 |
| 2024-07-03     | 0.45987541223891537   | 157.0                | 158.5             | 156.7            | 0.19955771016637025 | 1.47%                   | 0.4582900400771857  | 0.4784566162806291 | 0.4489477067332708 |
| 2024-07-02     | 0.4514474166361304    | 157.0                | 157.1             | 155.3            | 0.1963939999685203  | 0.00%                   | 0.45806738904556926 | 0.477651972335575  | 0.3822733120522838 |
| 2024-07-01     | 0.4514474166361304    | 154.1                | 156.4             | 153.2            | 0.21765066973069114 | 2.90%                   | 0.45947751224580685 | 0.47717301760637587 | 0.43011868766641087 |
| 2024-06-28     | 0.4353242946134116    | 151.9                | 153.5             | 151.6            | 0.1845261517636504  | 0.33%                   | 0.45643461481371533 | 0.47565952066210704 | 0.33724064682331134 |
| 2024-06-27     | 0.4334921216562845    | 150.4                | 151.3             | 149.7            | 0.15839012796499458 | 0.00%                   | 0.45175894314977    | 0.4743950801770217  | 0.4192584424426622 |
| 2024-06-26     | 0.4334921216562845    | 151.0                | 151.4             | 149.3            | 0.15932664914296507 | 0.27%                   | 0.4464153183909752  | 0.4730923233136004  | 0.43011868766641087 |
| 2024-06-25     | 0.43202638329058274   | 149.4                | 150.9             | 148.6            | 0.15853178663056994 | 1.00%                   | 0.442481816832418   | 0.4716363009368355  | 0.4214017796692396 |
| 2024-06-24     | 0.42652986441920127   | 148.0                | 149.5             | 147.5            | 0.16063305683660462 | 1.63%                   | 0.43743506011577854 | 0.46947142556085597 | 0.39734363349215585 |



### 1. Exploratory Data Analysis (EDA)

|      | 日付け (Date)                 | 終値 (Closing Price) | 始値 (Opening Price) | 高値 (High Price) |
|------|-------------------------------|----------------------|----------------------|-------------------|
| count| 9202                          | 9202.000000          | 9202.000000          | 9202.000000       |
| mean | 2005-10-21 08:44:04.642469248 | 92.180961            | 92.256183            | 93.176451         |
| min  | 1987-02-12 00:00:00           | 33.000000            | 33.000000            | 33.200000         |
| 25%  | 1996-06-06 06:00:00           | 52.000000            | 52.100000            | 52.800000         |
| 50%  | 2005-10-11 12:00:00           | 85.100000            | 85.100000            | 86.050000         |
| 75%  | 2015-03-04 18:00:00           | 110.800000           | 110.800000           | 111.900000        |
| max  | 2024-08-01 00:00:00           | 305.900000           | 309.800000           | 311.800000        |
| std  | NaN                           | 50.452228            | 50.598215            | 51.049837         |

|      | 安値 (Low Price) |
|------|------------------|
| count| 9202.000000      |
| mean | 91.330146        |
| min  | 32.200000        |
| 25%  | 51.500000        |
| 50%  | 84.200000        |
| 75%  | 109.275000       |
| max  | 303.900000       |
| std  | 50.087405        |




![image](https://github.com/user-attachments/assets/88db715f-49aa-4d09-bc99-e55c85be99b2)

### 2. Preprocessing and Feature Engineering (handling missing values, scaling, feature creation):
- Training set size: 7346
- Test set size: 1837
- Features used: ['終値', '始値', '高値', '安値', '出来高', 'MA_5', 'MA_20', 'Volatility', 'PriceRange', '変化率 %', 'DayOfWeek', 'Month']

### Modal Result:

- Modal Traning size = 0.8

- Mean Squared Error (MSE): 2.01
- Root Mean Squared Error (RMSE): 1.42
- Mean Absolute Error (MAE): 1.15
- R-squared (R²): 0.9964
- Model Accuracy: 99.64%

![predictions_vs_actuals_train_size0 8](https://github.com/user-attachments/assets/52dd8c41-e692-48fd-b8d0-879ece79e240)

 - ### Actual vs Predicted Stock Prices

| Date       | Actual Prices | Predicted Prices | Error               |
|------------|---------------|------------------|---------------------|
| 2017-01-27 | 99.5          | 99.28466076105832| 0.21533923894168083  |
| 2017-01-30 | 98.9          | 98.0040260463953 | 0.8959739536046953   |
| 2017-01-31 | 99.6          | 99.15734593719243| 0.4426540628075628   |
| 2017-02-01 | 99.2          | 98.25687455534934| 0.9431254446506472   |
| 2017-02-02 | 97.2          | 96.88484871536492| 0.31515128463506414  |
| 2017-02-03 | 97.5          | 97.57405589222907| -0.07405589222908304 |
| 2017-02-06 | 97.0          | 96.73309813141822| 0.266901868581769    |
| 2017-02-07 | 97.5          | 96.85278821736573| 0.6472117826342583   |
| 2017-02-08 | 96.6          | 96.17720390558242| 0.4227960944175635   |
| 2017-02-09 | 97.3          | 97.03295569121836| 0.2670443087816068   |
| 2017-02-10 | 99.1          | 98.84321486800908| 0.2567851319909096   |
| 2017-02-13 | 97.4          | 96.68403548449277| 0.7159645155072383   |
| 2017-02-14 | 95.2          | 95.52512818574904| -0.3251281857490511  |
| 2017-02-15 | 95.3          | 95.2266288459301 | 0.07337115406988914  |
| 2017-02-16 | 94.3          | 94.56623707860707| -0.2662370786070767  |
| 2017-02-17 | 94.2          | 94.13774714022874| 0.06225285977126305  |
| 2017-02-20 | 95.4          | 94.62940244078635| 0.7705975592136411   |
| 2017-02-21 | 94.5          | 94.03876787573098| 0.46123212426901716  |
| 2017-02-22 | 94.7          | 94.32909756302833| 0.3709024369716474   |




