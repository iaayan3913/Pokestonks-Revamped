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


model = ARIMA(df['PRICE'], order=(4, 0, 5))
model = model.fit()
best_aic = model.aic
best_order = (4, 0, 5)
best_model = model



# === Model diagnostics ===
def analyze_model_quality(model_fit):
    print(f"\n=== MODEL QUALITY ASSESSMENT ===")
    
    # Get residuals
    residuals = model_fit.resid.dropna()
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    axes[0,0].plot(residuals, linewidth=1, color='red')
    axes[0,0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    
    # ACF of residuals
    plot_acf(residuals, ax=axes[1,0], lags=20)
    axes[1,0].set_title('Residual Autocorrelation', fontsize=14, fontweight='bold')
    
    # Residuals histogram
    axes[1,1].hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    
    # Add normal curve
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
    
    print(f"📊 Model Diagnostics:")
    print(f"   Mean residual: {residuals.mean():.4f} (should be near 0)")
    print(f"   Residual std: {residuals.std():.2f}")
    
    # Check Ljung-Box test results
    significant_lags = ljung_box[ljung_box['lb_pvalue'] <= 0.05]
    if len(significant_lags) == 0:
        print(f"   ✅ Ljung-Box test: PASSED (no significant autocorrelation in residuals)")
        print(ljung_box[['lb_pvalue']])
    else:
        print(f"   ⚠️  Ljung-Box test: Some autocorrelation detected at {len(significant_lags)} lags")
    
    return residuals

print(f"\n🔍 Analyzing model quality...")
residuals = analyze_model_quality(best_model)

""" residuals over time constantly fluctuate around 0, which is good
the Q-Q plot shows that the residuals are all close to a y=z line, which indicates normality 
all the points are within the confidence interval. Lag 0 is always at 1 so fits the ideal graph
the plot of the histrogram does not show a perfect normal distribution, however, it does resemble the graph relatively closely which is a good sign
all the probabiltiies from the ljung box test are 0.5 which is a sign to reject the null hypothesis of autocorrelation in the residuals hence test can be passed
"""

# === 6. Model diagnostics ===
def analyze_model_quality(model_fit):
    print(f"\n=== MODEL QUALITY ASSESSMENT ===")
    
    # Get residuals
    residuals = model_fit.resid.dropna()
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals over time
    axes[0,0].plot(residuals, linewidth=1, color='red')
    axes[0,0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    
    # ACF of residuals
    plot_acf(residuals, ax=axes[1,0], lags=20)
    axes[1,0].set_title('Residual Autocorrelation', fontsize=14, fontweight='bold')
    
    # Residuals histogram
    axes[1,1].hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    
    # Add normal curve
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
    
    print(f"📊 Model Diagnostics:")
    print(f"   Mean residual: {residuals.mean():.4f} (should be near 0)")
    print(f"   Residual std: {residuals.std():.2f}")
    
    # Check Ljung-Box test results
    significant_lags = ljung_box[ljung_box['lb_pvalue'] <= 0.05]
    if len(significant_lags) == 0:
        print(f"   ✅ Ljung-Box test: PASSED (no significant autocorrelation in residuals)")
    else:
        print(f"   ⚠️  Ljung-Box test: Some autocorrelation detected at {len(significant_lags)} lags")
    
    return residuals

print(f"\n🔍 Analyzing model quality...")
residuals = analyze_model_quality(best_model)




# === 7. Walk-forward validation ===
def validate_model(data, model_order, test_weeks=8):
    print(f"\n=== MODEL VALIDATION ===")
    
    if len(data) < test_weeks + 10:
        test_weeks = max(4, len(data) // 6)
        print(f"Adjusting validation period to {test_weeks} weeks due to limited data")
    
    train_data = data[:-test_weeks]
    test_data = data[-test_weeks:]
    
    predictions = []
    actuals = list(test_data['PRICE'])
    successful_predictions = 0
    
    print(f"Testing model on last {test_weeks} weeks:")
    print("-" * 60)
    
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
            
            print(f"Week {i+1}: {status} ${pred:.2f} vs ${actuals[i]:.2f} (error: {pct_error:.1f}%)")
            
        except Exception as e:
            print(f"Week {i+1}: ❌ Model failed - {str(e)[:50]}...")
            # For failed predictions, use a simple average of recent prices
            recent_avg = current_train['PRICE'].tail(4).mean()
            predictions.append(recent_avg)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
    
    print(f"\n📈 Validation Results:")
    print(f"   MAE:  ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.1f}%")
    
    # Accuracy rating
    if mape < 5:
        rating = "🎯 Excellent"
    elif mape < 10:
        rating = "✅ Very Good"
    elif mape < 15:
        rating = "👍 Good"
    elif mape < 25:
        rating = "⚠️  Fair"
    else:
        rating = "❌ Poor"
    
    print(f"   Accuracy: {rating}")
    
    return mape, mae, rmse

# Run validation
mape, mae, rmse = validate_model(df, model)

# === 8. Generate forecast ===
forecast_weeks = 6
print(f"\n=== GENERATING {forecast_weeks}-WEEK FORECAST ===")

forecast_result = model.get_forecast(steps=forecast_weeks)
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
ax1.set_title(f'Pokemon Card Price Forecast - ARIMA{best_order}', fontsize=16, fontweight='bold')
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
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Price ($)', fontsize=12)
ax2.set_title(f'Recent Trend & {forecast_weeks}-Week Forecast', fontsize=16, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === 10. Detailed forecast results ===
print(f"\n🎯 DETAILED FORECAST RESULTS")
print("=" * 60)

current_price = df['PRICE'].iloc[-1]
final_forecast = forecast.iloc[-1]
total_change = final_forecast - current_price
total_change_pct = (total_change / current_price) * 100

for i, (date, price, lower, upper) in enumerate(zip(
    forecast_dates, forecast, conf_int.iloc[:, 0], conf_int.iloc[:, 1])):
    
    week_change = price - current_price
    week_change_pct = (week_change / current_price) * 100
    uncertainty = (upper - lower) / 2
    
    print(f"Week {i+1} ({date.strftime('%Y-%m-%d')}):")
    print(f"  📈 Predicted: ${price:.2f} ({week_change_pct:+.1f}% from current)")
    print(f"  📊 Range: ${lower:.2f} - ${upper:.2f} (±${uncertainty:.2f})")
    print()

# === 11. Summary and recommendations ===
print("🎯 EXECUTIVE SUMMARY")
print("=" * 60)
print(f"📊 Model: ARIMA{best_order} (AIC: {best_aic:.1f})")
print(f"🎪 Validation Accuracy: {mape:.1f}% MAPE")
print(f"💰 Current Price: ${current_price:.2f}")
print(f"🔮 6-Week Forecast: ${final_forecast:.2f}")
print(f"📈 Expected Change: ${total_change:+.2f} ({total_change_pct:+.1f}%)")

if total_change_pct > 5:
    trend = "📈 Strong Upward Trend"
elif total_change_pct > 1:
    trend = "↗️  Mild Upward Trend"
elif total_change_pct > -1:
    trend = "➡️  Stable/Sideways"
elif total_change_pct > -5:
    trend = "↘️  Mild Downward Trend"
else:
    trend = "📉 Strong Downward Trend"

print(f"📊 Trend: {trend}")

avg_uncertainty = np.mean((conf_int.iloc[:, 1] - conf_int.iloc[:, 0]) / 2)
uncertainty_pct = (avg_uncertainty / final_forecast) * 100

if uncertainty_pct < 5:
    confidence = "🎯 High Confidence"
elif uncertainty_pct < 10:
    confidence = "✅ Good Confidence"  
elif uncertainty_pct < 15:
    confidence = "⚠️  Moderate Confidence"
else:
    confidence = "❓ Low Confidence"

print(f"🎪 Confidence Level: {confidence} (±{uncertainty_pct:.1f}%)")

print(f"\n💡 RECOMMENDATIONS:")
if mape < 10:
    print(f"✅ Model shows good accuracy - forecasts are reliable for planning")
else:
    print(f"⚠️  Model accuracy is moderate - use forecasts as general guidance")



print(f"\n" + "=" * 60)
print(f"🎉 ANALYSIS COMPLETE!")
print(f"=" * 60)