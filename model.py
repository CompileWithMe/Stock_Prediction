import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Use closing price
data['Prediction'] = data[['Close']].shift(-1)

X = data[['Close']][:-1]
y = data['Prediction'][:-1]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'models/stock_model.pkl')

print("Model Created Successfully")