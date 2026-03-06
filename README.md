# Bitcoin Price Prediction using Linear Regression

Live App: https://yqhhc2pgpyktsjbvy3cbli.streamlit.app

This project predicts Bitcoin prices using historical data and Linear Regression.

## Features
- Upload Bitcoin dataset
- Train machine learning model
- Predict future price
- Visualization using Matplotlib


📈 Bitcoin Price Prediction using Linear Regression

This project builds a simple machine learning model to predict the next day's Bitcoin closing price using historical daily trading data. It involves data preprocessing, feature engineering, and training a linear regression model. The performance is evaluated using common regression metrics and visualized with Matplotlib.

📂 Dataset

The dataset consists of daily Bitcoin trading data with the following features:

Open Time: Timestamp of the trading day start

Open: Opening price

High: Highest price of the day

Low: Lowest price of the day

Close: Closing price (used as target)

Volume: Trading volume for the day


📥 Download Sample CSV from CryptoDataDownload

🔧 Features Used

The following features were selected for prediction:

Open

High

Low

Volume

Target: Next day's Close price


🚀 Workflow

1. Load & Preprocess the Data
Convert Open Time to datetime

Sort by date

Create Target column by shifting Close one day up

Drop last row with NaN

2. Train-Test Split
Use 80% for training, 20% for testing (no shuffle to maintain time sequence)

3. Train Model
python
Copy
Edit


from sklearn.linear_model import LinearRegression
5. Evaluate Model
Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

5. Visualize
6. 
Actual vs Predicted prices plotted using Matplotlib


📊 Evaluation Example

makefile
Copy
Edit
MAE: 312.25
RMSE: 408.33
R²: 0.8724


📌 Requirements

Python 3.x

pandas

numpy

matplotlib

scikit-learn

📸 Sample Output Plot

(optional if you generate a plot image)


✍️ Author

Alamgir Khan
📘 GitHub Profile

