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
