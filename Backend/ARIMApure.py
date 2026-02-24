import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("svp48test1.csv")
# uk dates use dd/mm/yyyy format
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
# Sort by date and remove duplicates
df = df.sort_values('DATE').drop_duplicates(subset=['DATE'], keep='first')
df.set_index('DATE', inplace=True)
# Convert price to numeric and handle missing values
df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
if df['PRICE'].isna().sum() > 0:
    df['PRICE'] = df['PRICE'].interpolate(method='linear')
    
    # use a brute force apprach to find the best ARIMA model
def find_best_arima(data, max_p=5, max_d=2, max_q=5):
    
    # Adjust search space based on data size
    if len(data) < 50:
        max_p, max_d, max_q = 4, 2, 4
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    models_tested = 0
    successful_models = 0
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    
                    if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                    
                    successful_models += 1
                except:
                    pass
                finally:
                    models_tested += 1
                    
                # Progress indicator
                if models_tested % 20 == 0:
                    progress = (models_tested / total_combinations) * 100
    return best_model, best_order, best_aic

# Find best model
best_model, best_order, best_aic = find_best_arima(df['PRICE'])


# === 7. Walk-forward validation ===
def validate_model(data, model_order, test_weeks=8):
    
    if len(data) < test_weeks + 10:
        test_weeks = max(4, len(data) // 6)

    
    train_data = data[:-test_weeks]
    test_data = data[-test_weeks:]
    
    predictions = []
    actuals = list(test_data['PRICE'])
    successful_predictions = 0
    

    
    for i in range(test_weeks):
        current_train = data[:len(train_data) + i]
        
        try:
            model = ARIMA(current_train['PRICE'], order=model_order)
            fitted_model = model.fit()
            pred = fitted_model.forecast(steps=1)[0]
            predictions.append(pred)
            successful_predictions += 1
            
            error = abs(pred - actuals[i])
            pct_error = (error / actuals[i]) * 100
            
            # More lenient success criteria
            status = "✅" if pct_error < 15 else "⚠️" if pct_error < 25 else "❌"
            
            #print(f"Week {i+1}: {status} ${pred:.2f} vs ${actuals[i]:.2f} (error: {pct_error:.1f}%)")
            
        except Exception as e:
            #print(f"Week {i+1}: ❌ Model failed - {str(e)[:50]}...")
            # For failed predictions, use a simple average of recent prices
            recent_avg = current_train['PRICE'].tail(4).mean()
            predictions.append(recent_avg)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
    

    

    
    return mape, mae, rmse

# Run validation
mape, mae, rmse = validate_model(df, best_model)

# === 8. Generate forecast ===
forecast_weeks = 6

forecast_result = best_model.get_forecast(steps=forecast_weeks)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Create forecast dates
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), 
                              periods=forecast_weeks, freq='W')

# === 9. Visualization ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Full time series
ax1.plot(df.index, df['PRICE'], label='Historical Prices', linewidth=2.5, color='#2E86AB', alpha=0.8)
ax1.plot(forecast_dates, forecast, label='ARIMA Forecast', 
         linewidth=3, color='#F24236', marker='o', markersize=6)
ax1.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                color='#F24236', alpha=0.2, label='95% Confidence Interval')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.set_title(f'Pokemon Card Price Forecast - ARIMA{best_order}', fontsize=12, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Recent data focus
recent_weeks = min(20, len(df))
recent_data = df.tail(recent_weeks)
ax2.plot(recent_data.index, recent_data['PRICE'], 
         label='Recent Prices', linewidth=2.5, color='#2E86AB', marker='o', markersize=4, alpha=0.8)
ax2.plot(forecast_dates, forecast, label='ARIMA Forecast', 
         linewidth=3, color='#F24236', marker='o', markersize=6)
ax2.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                color='#F24236', alpha=0.2, label='95% Confidence Interval')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title(f'Recent Trend & {forecast_weeks}-Week Forecast', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()










