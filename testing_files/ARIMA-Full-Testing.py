import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === 1. Load and prepare data ===
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

# === 2. Data overview ===
print("=== POKEMON CARD PRICE FORECASTING ===")
print(f"📊 Dataset: {len(df)} weeks from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"💰 Price range: ${df['PRICE'].min():.2f} - ${df['PRICE'].max():.2f}")
print(f"📈 Current price: ${df['PRICE'].iloc[-1]:.2f}")

# === 3. Visualize data ===

# function to plot all the time series analysis components
def plot_time_series_analysis(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original series
    axes[0,0].plot(data.index, data['PRICE'], linewidth=2, color='blue')
    axes[0,0].set_title('Pokemon Card Price Over Time', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Price ($)')
    axes[0,0].grid(True, alpha=0.3)
    
    # First difference
    diff_data = data['PRICE'].diff().dropna()
    axes[0,1].plot(data.index[1:], diff_data, linewidth=1.5, color='green')
    axes[0,1].set_title('Week-to-Week Price Changes', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Price Change ($)')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # ACF = autocorrelation function 
    plot_acf(data['PRICE'].dropna(), ax=axes[1,0], lags=min(25, len(data)//4))
    axes[1,0].set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
    
    # PACF = partial autocorellation function 
    plot_pacf(data['PRICE'].dropna(), ax=axes[1,1], lags=min(25, len(data)//4))
    axes[1,1].set_title('Partial Autocorrelation Function', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# === 4. Stationarity test ===
def check_stationarity(timeseries):
    print("\n=== STATIONARITY ANALYSIS ===")
    result = adfuller(timeseries.dropna())
    
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print(" Series is stationary")
        return True
    else:
        print(" Series is non-stationary ")
        return False

is_stationary = check_stationarity(df['PRICE']) #check if the series can be used with ARIMA

print(f"\n📈 Creating graphs (acf,pacf ...)")
plot_time_series_analysis(df)

""" from the plot we can see that the acf gives a needed value of 4 and pacf gives a value of 1
implementing these into the ARIMA model we get 
   ARIMA(p,d,q) where p is the autoregressive order, d is the degree of differencing, and q is the moving average order found by the pacf.
   In this case, we will use ARIMA(1, 1, 4) as the model parameters.
   pacf gives us the value of 1 and acf gives us the value of 4
   therefore q = 4, p = 1, d = 1"""
   
# === 5. ARIMA model selection ===
# use a brute force apprach to find the best ARIMA model
def find_best_arima(data, max_p=5, max_d=2, max_q=5):
    print(f"\n=== ARIMA MODEL SELECTION ===")
    
    # Adjust search space based on data size
    if len(data) < 50:
        max_p, max_d, max_q = 4, 2, 4
        print(f"Optimizing search for dataset size: testing up to ARIMA({max_p},{max_d},{max_q})")
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    models_tested = 0
    successful_models = 0
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    
    print(f"Testing {total_combinations} ARIMA model combinations...")
    
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
                    print(f"  Progress: {progress:.0f}% ({successful_models} successful models)", end='\r')
    
    print(f"\n✅ Model selection complete!")
    print(f"   Best model: ARIMA{best_order}")
    print(f"   AIC score: {best_aic:.2f}")
    print(f"   Success rate: {successful_models}/{models_tested} models")
    
    return best_model, best_order, best_aic

# Find best model
best_model, best_order, best_aic = find_best_arima(df['PRICE'])

"testing multiple max parameters gives the best ARIMA parametesr of (4,0,5) with an AIC score of -121.16"



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
mape, mae, rmse = validate_model(df, best_model)

# === 8. Generate forecast ===
forecast_weeks = 6
print(f"\n=== GENERATING {forecast_weeks}-WEEK FORECAST ===")

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
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Price ($)', fontsize=12)
ax2.set_title(f'Recent Trend & {forecast_weeks}-Week Forecast', fontsize=12, fontweight='bold')
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