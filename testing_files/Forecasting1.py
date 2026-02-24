import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv('testdata1.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df[["date","card_price(price_charting)"]]
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])

print(f"Dataset size: {len(df)} observations")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

# Create smooth interpolation between monthly points
def create_smooth_historical_curve(df):
    """Create smooth daily interpolation of monthly data"""
    
    # Convert dates to numeric for interpolation
    df_numeric = df.copy()
    df_numeric['ds_numeric'] = (df_numeric['ds'] - df_numeric['ds'].min()).dt.days
    
    # Create spline interpolation
    tck = interpolate.splrep(df_numeric['ds_numeric'], df_numeric['y'], s=0, k=min(3, len(df)-1))
    
    # Create daily points between first and last date
    start_date = df['ds'].min()
    end_date = df['ds'].max()
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_numeric = (daily_dates - start_date).days
    
    # Interpolate daily values
    daily_values = interpolate.splev(daily_numeric, tck)
    
    # Apply slight smoothing to remove any artifacts
    daily_values = gaussian_filter1d(daily_values, sigma=1.5)
    
    smooth_df = pd.DataFrame({
        'ds': daily_dates,
        'y': daily_values
    })
    
    return smooth_df

# Method 1: Polynomial Trend Forecasting (captures curves)
def polynomial_forecast_daily(df, degree=3, days_ahead=90):
    """Fit polynomial to capture curves, then forecast daily"""
    
    # Convert to numeric for polynomial fitting
    df_numeric = df.copy()
    df_numeric['ds_numeric'] = (df_numeric['ds'] - df_numeric['ds'].min()).dt.days
    
    X = df_numeric['ds_numeric'].values.reshape(-1, 1)
    y = df_numeric['y'].values
    
    # Fit polynomial
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Create future daily predictions
    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=days_ahead, freq='D')
    
    future_numeric = (future_dates - df['ds'].min()).days.values.reshape(-1, 1)
    future_poly = poly_features.transform(future_numeric)
    predictions = model.predict(future_poly)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'method': f'Polynomial Degree {degree}'
    })
    
    return model, poly_features, forecast_df

# Method 2: Spline Extrapolation
def spline_forecast_daily(df, days_ahead=90):
    """Use spline fitting and extrapolation"""
    
    df_numeric = df.copy()
    df_numeric['ds_numeric'] = (df_numeric['ds'] - df_numeric['ds'].min()).dt.days
    
    # Fit spline
    tck = interpolate.splrep(df_numeric['ds_numeric'], df_numeric['y'], s=0, k=min(3, len(df)-1))
    
    # Extrapolate (note: spline extrapolation can be unstable)
    last_day = df_numeric['ds_numeric'].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1)
    
    # For stability, limit extrapolation and add trend
    try:
        predictions = interpolate.splev(future_days, tck)
        
        # If extrapolation goes wild, apply constraints
        last_value = df['y'].iloc[-1]
        recent_trend = df['y'].iloc[-3:].diff().mean() if len(df) >= 3 else 0
        
        # Cap predictions if they deviate too much
        for i, pred in enumerate(predictions):
            expected_range = last_value + recent_trend * (i + 1) / 30.44  # monthly trend to daily
            if abs(pred - expected_range) > abs(last_value * 0.5):  # If prediction is >50% off
                predictions[i] = expected_range
                
    except:
        # Fallback to linear trend if spline fails
        recent_trend = df['y'].diff().mean()
        daily_trend = recent_trend / 30.44
        predictions = [df['y'].iloc[-1] + daily_trend * (i + 1) for i in range(days_ahead)]
    
    future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(days=1), 
                                periods=days_ahead, freq='D')
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'method': 'Spline Extrapolation'
    })
    
    return forecast_df

# Method 3: Exponential Smoothing with Curve Fitting
def exponential_smooth_forecast(df, days_ahead=90, alpha=0.3):
    """Apply exponential smoothing and project trend"""
    
    values = df['y'].values
    
    # Double exponential smoothing (Holt's method)
    smoothed = [values[0]]
    trend = [values[1] - values[0] if len(values) > 1 else 0]
    
    for i in range(1, len(values)):
        if i == 1:
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[0])
            trend.append(alpha * (smoothed[1] - smoothed[0]) + (1 - alpha) * trend[0])
        else:
            new_smooth = alpha * values[i] + (1 - alpha) * (smoothed[i-1] + trend[i-1])
            new_trend = alpha * (new_smooth - smoothed[i-1]) + (1 - alpha) * trend[i-1]
            smoothed.append(new_smooth)
            trend.append(new_trend)
    
    # Project forward
    last_smooth = smoothed[-1]
    last_trend = trend[-1]
    daily_trend = last_trend / 30.44  # Convert monthly to daily
    
    predictions = []
    for i in range(days_ahead):
        pred = last_smooth + daily_trend * (i + 1)
        predictions.append(pred)
    
    future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(days=1), 
                                periods=days_ahead, freq='D')
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': predictions,
        'method': 'Exponential Smoothing'
    })
    
    return forecast_df

# Create smooth historical curve
print("Creating smooth interpolation of historical data...")
smooth_historical = create_smooth_historical_curve(df)

# Run forecasting methods
print("Running curved forecasting methods...")
print("="*50)

# Polynomial forecasts with different degrees
poly2_model, poly2_features, poly2_forecast = polynomial_forecast_daily(df, degree=2, days_ahead=90)
poly3_model, poly3_features, poly3_forecast = polynomial_forecast_daily(df, degree=3, days_ahead=90)

# Spline extrapolation
spline_forecast = spline_forecast_daily(df, days_ahead=90)

# Exponential smoothing
exp_smooth_forecast = exponential_smooth_forecast(df, days_ahead=90)

# Combine forecasts
all_forecasts = pd.concat([
    poly2_forecast,
    poly3_forecast,
    spline_forecast,
    exp_smooth_forecast
], ignore_index=True)

# Create comprehensive plot
plt.figure(figsize=(20, 12))

# Plot 1: Historical data comparison (original vs smooth)
plt.subplot(2, 2, 1)
plt.plot(df['ds'], df['y'], 'ro-', label='Original Monthly Points', linewidth=2, markersize=6)
plt.plot(smooth_historical['ds'], smooth_historical['y'], 'b-', label='Smooth Interpolation', linewidth=2, alpha=0.7)
plt.title('Historical Data: Original vs Smooth Interpolation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Focus on recent data and short-term forecast (30 days)
plt.subplot(2, 2, 2)
recent_data = smooth_historical.tail(90) if len(smooth_historical) > 90 else smooth_historical
plt.plot(recent_data['ds'], recent_data['y'], 'b-', label='Recent Historical (Smooth)', linewidth=2)
plt.plot(df['ds'], df['y'], 'ro', label='Monthly Data Points', markersize=6)

# Show 30-day forecasts
methods = all_forecasts['method'].unique()
colors = ['red', 'green', 'orange', 'purple']

for i, method in enumerate(methods):
    method_data = all_forecasts[all_forecasts['method'] == method].head(30)
    plt.plot(method_data['ds'], method_data['yhat'], 
             color=colors[i], label=f'{method} (30d)', linewidth=2, linestyle='--')

plt.title('Recent Data + 30-Day Forecasts')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Full 90-day forecasts
plt.subplot(2, 2, 3)
plt.plot(smooth_historical['ds'], smooth_historical['y'], 'b-', label='Historical (Smooth)', linewidth=2)
plt.plot(df['ds'], df['y'], 'ro', label='Monthly Data Points', markersize=6)

for i, method in enumerate(methods):
    method_data = all_forecasts[all_forecasts['method'] == method]
    plt.plot(method_data['ds'], method_data['yhat'], 
             color=colors[i], label=f'{method}', linewidth=2)

plt.title('Complete Historical + 90-Day Forecasts')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Forecast comparison only
plt.subplot(2, 2, 4)
for i, method in enumerate(methods):
    method_data = all_forecasts[all_forecasts['method'] == method]
    plt.plot(method_data['ds'], method_data['yhat'], 
             color=colors[i], label=f'{method}', linewidth=2)

# Add last historical point for context
plt.plot(df['ds'].iloc[-1], df['y'].iloc[-1], 'ko', markersize=8, label='Last Known Price')

plt.title('90-Day Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print forecast summary
print("\nCurved Forecast Summary (Key Dates):")
print("="*60)
print(f"Last known price: ${df['y'].iloc[-1]:.2f}")
print()

for method in methods:
    method_data = all_forecasts[all_forecasts['method'] == method]
    day7 = method_data.iloc[6]['yhat'] if len(method_data) >= 7 else None
    day30 = method_data.iloc[29]['yhat'] if len(method_data) >= 30 else None
    day90 = method_data.iloc[89]['yhat'] if len(method_data) >= 90 else None
    
    print(f"{method}:")
    if day7: print(f"  Day 7:  ${day7:.2f}")
    if day30: print(f"  Day 30: ${day30:.2f}")
    if day90: print(f"  Day 90: ${day90:.2f}")
    print()

print("Curve-Fitting Method Recommendations:")
print("="*60)
print("• POLYNOMIAL DEGREE 2: Good for smooth trends, conservative")
print("• POLYNOMIAL DEGREE 3: Captures more complex curves but can be volatile")
print("• SPLINE EXTRAPOLATION: Most faithful to historical curves but risky for long forecasts")
print("• EXPONENTIAL SMOOTHING: Balances recent trends with stability")
print("\nFor card prices: Polynomial Degree 2 is usually most reliable.")