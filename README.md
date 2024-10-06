# Stock Price Prediction using Time-Series Data

## Project Overview

This project involves building a stock price prediction model using NTT stock data. The steps include Exploratory Data Analysis (EDA), Data Preprocessing, Model Selection, Model Evaluation, and Model Improvement.

---
## Setup and Installation
### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

## Step 2: Create and Activate the Virtual Environment
### For Windows:
```bash
python -m venv env
env\Scripts\activate
```


## For Linux/macOS:
``` bash
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
Copy code
pip install -r requirements.txt
```

## Train the Model
### After performing EDA and preprocessing, run the model training script:

```bash
python models/model.py
```

## Results:

## 1. Exploratory Data Analysis (EDA)

| 日付け (Date)           | 終値 (Closing Price) | 始値 (Opening Price) | 高値 (High Price) | 安値 (Low Price) |
|------------------------|----------------------|----------------------|-------------------|-----------------|
| count                  | 9202                 | 9202.000000           | 9202.000000        | 9202.000000      |
| mean                   | 2005-10-21 08:44:04.642469248 | 92.180961 | 92.256183 | 93.176451 |
| min                    | 1987-02-12 00:00:00  | 33.000000             | 33.000000          | 33.200000        |
| 25%                    | 1996-06-06 06:00:00  | 52.000000             | 52.100000          | 52.800000        |
| 50%                    | 2005-10-11 12:00:00  | 85.100000             | 85.100000          | 86.050000        |
| 75%                    | 2015-03-04 18:00:00  | 110.800000            | 110.800000         | 111.900000       |
| max                    | 2024-08-01 00:00:00  | 305.900000            | 309.800000         | 311.800000       |
| std                    | NaN                  | 50.452228             | 50.598215          | 51.049837        |

### 安値 (Low Price) Statistics
| Count | Mean     | Min      | 25%      | 50%      | 75%      | Max      | Std       |
|-------|----------|----------|----------|----------|----------|----------|-----------|
| 9202  | 91.330146 | 32.200000 | 51.500000 | 84.200000 | 109.275000 | 303.900000 | 50.087405 |



![image](https://github.com/user-attachments/assets/88db715f-49aa-4d09-bc99-e55c85be99b2)

## 2. Preprocessing and Feature Engineering (handling missing values, scaling, feature creation):
- Converted String Numbers to Floats.
- Handled Units: If our dataset has values in millions (e.g., 220.24M), need to convert them into a standard numeric format.

  - ### Output:
## Stock Price Data

| 日付け (Date)      | 終値 (Closing Price) | 始値 (Opening Price) | 高値 (High Price) | 安値 (Low Price) | 出来高 (Volume) | 変化率 % (Change Rate %) | MA_5     | MA_20    | Pct_change |
|--------------------|----------------------|----------------------|-------------------|------------------|-----------------|-------------------------|----------|----------|------------|
| 2024-07-04         | 0.4565775009160865   | 159.0                | 159.1             | 157.2            | 0.16597673649914219 | -0.57%                  | 0.45814160605610793 | 0.478035136118934 | 0.4660724385636594 |
| 2024-07-03         | 0.45987541223891537  | 157.0                | 158.5             | 156.7            | 0.19955771016637025 | 1.47%                   | 0.4582900400771857  | 0.4784566162806291 | 0.4489477067332708 |
| 2024-07-02         | 0.4514474166361304   | 157.0                | 157.1             | 155.3            | 0.1963939999685203  | 0.00%                   | 0.45806738904556926 | 0.477651972335575  | 0.3822733120522838 |
| 2024-07-01         | 0.4514474166361304   | 154.1                | 156.4             | 153.2            | 0.21765066973069114 | 2.90%                   | 0.45947751224580685 | 0.47717301760637587 | 0.43011868766641087 |
| 2024-06-28         | 0.4353242946134116   | 151.9                | 153.5             | 151.6            | 0.1845261517636504  | 0.33%                   | 0.45643461481371533 | 0.47565952066210704 | 0.33724064682331134 |
| 2024-06-27         | 0.4334921216562845   | 150.4                | 151.3             | 149.7            | 0.15839012796499458 | 0.00%                   | 0.45175894314977    | 0.4743950801770217  | 0.4192584424426622 |
| 2024-06-26         | 0.4334921216562845   | 151.0                | 151.4             | 149.3            | 0.15932664914296507 | 0.27%                   | 0.4464153183909752  | 0.4730923233136004  | 0.43011868766641087 |
| 2024-06-25         | 0.43202638329058274  | 149.4                | 150.9             | 148.6            | 0.15853178663056994 | 1.00%                   | 0.442481816832418   | 0.4716363009368355  | 0.4214017796692396 |
| 2024-06-24         | 0.42652986441920127  | 148.0                | 149.5             | 147.5            | 0.16063305683660462 | 1.63%                   | 0.43743506011577854 | 0.46947142556085597 | 0.39734363349215585 |

## Modal Result Before Improvement:

- Mean Squared Error (MSE): 0.0004651555017138786
- Root Mean Squared Error (RMSE): 0.021567463961112317
- Mean Absolute Error (MAE): 0.013755883415600337
- R-squared (R²): 0.9862914681434631
- Model Accuracy: 98.62914681434631

![predictions_vs_actuals](https://github.com/user-attachments/assets/dd4ae8de-b503-417d-9fc9-d2afad7a0b83)

## Modal Result After Improvement:

--- Results ---

- **Best MSE**: 4.869792779494177e-06
- **Best parameters**: 
  - LSTM Units: 200
  - Dense Units: 50
  - Dropout Rate: 0.1
  - Learning Rate: 0.0005

- **Mean Squared Error (MSE)**: 2.9467505371240005e-06
- **Root Mean Squared Error (RMSE)**: 0.0017166101878772597
- **Mean Absolute Error (MAE)**: 0.0012187551153146244
- **R-squared (R²)**: 0.9999131560325623
- **Model Accuracy**: 99.99131560325623%

![Final_output](https://github.com/user-attachments/assets/a3cede78-3095-4fd4-bbfe-7368d7591ce1)



