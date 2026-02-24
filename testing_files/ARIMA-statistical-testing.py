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
