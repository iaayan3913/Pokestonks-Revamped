import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# === 1. Load your data ===
df = pd.read_csv("svp48test1.csv")  # Change to your file name

# === 2. Convert DATE column to datetime ===
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)

# === 3. Set DATE as index ===
df.set_index('DATE', inplace=True)




# === 4. Make sure PRICE is numeric ===
df['PRICE'] = pd.to_numeric(df['PRICE'])

# === 5. Create ARIMA model ===
# (p=1, d=1, q=1) are common starting values; tweak later
model = ARIMA(df['PRICE'], order=(1, 1, 1))
model_fit = model.fit()

# === 6. Forecast next 6 months ===
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)

# === 7. Plot ===
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['PRICE'], label='Historical Prices')
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='M')[1:], 
         forecast, label='Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Pokemon Card Price Forecast (ARIMA)')
plt.legend()
plt.grid(True)
plt.show()

# === 8. Print forecast ===
print("Price forecast for next", forecast_steps, "months:")
print(forecast)
