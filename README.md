# 🎴 PokéTracker (Pokestonks)

# 

PokéTracker is an AI-powered Pokémon TCG market analyst and card viewer. It allows users to search for Pokémon cards, view real-time pricing data, forecast future market trends using machine learning, and chat with an integrated AI assistant for investment recommendations.

## ✨ Features

# 

-   **Comprehensive Card Search:** Search by Pokémon name or specific Card ID using the free, open-source TCGdex API.
    
-   **Live Pricing Data:** Automatically extracts and normalizes pricing data (Low, Mid, High, Market) from TCGplayer and Cardmarket.
    
-   **Machine Learning Price Forecasting:** Uses an ARIMA (AutoRegressive Integrated Moving Average) time-series model to predict future card values based on historical CSV data.
    
-   **AI Market Analyst:** Choose between Anthropic's Claude or Google's Gemini to get instant, context-aware advice. The AI can:
    
    -   Analyze price trend forecasts.
        
    -   Recommend similar cards for investment.
        
    -   Chat with you about general Pokémon TCG knowledge.
        
-   **Interactive Visualizations:** Renders historical data and future predictions (with 95% confidence intervals) using Chart.js.
    

## 🛠️ Tech Stack

# 

**Frontend:**

-   HTML5, CSS3, Vanilla JavaScript
    
-   [Chart.js](https://www.chartjs.org/) for price forecasting graphs
    
-   [Marked.js](https://marked.js.org/) for rendering AI markdown responses
    

**Backend:**

-   **Python 3.x**
    
-   **FastAPI** for routing and API endpoints
    
-   **Pandas & NumPy** for data manipulation
    
-   **Statsmodels** for ARIMA predictive modeling
    
-   **HTTPX** for async AI API calls
    

## 🚀 Installation & Setup

### 1\. Clone the Repository

# Bash

    git clone https://github.com/your-username/Pokestonks-Revamped.git
    cd Pokestonks-Revamped

### 2\. Backend Setup

# 

Make sure you have Python installed. It is highly recommended to create and activate a virtual environment first.

Bash

    # Install all required Python dependencies
    pip install -r requirements.txt

Create a `.env` file in the root directory and add your AI API keys:

Code snippet

    CLAUDE_API_KEY=your-claude-api-key-here
    GEMINI_API_KEY=your-gemini-api-key-here

_(Note: The app will still function if you only provide one of the keys, it will just disable the missing model.)_

Start the backend server:

Bash

    uvicorn app:app --reload

The backend will run on `http://127.0.0.1:8000`.

### 3\. Frontend Setup

# 

Because the frontend uses standard HTML/JS, you can simply open `Claude_website.html` in your browser. For the best development experience, use a local server like the VS Code "Live Server" extension.

## 📊 Data Requirements for Forecasting

# 

For the ARIMA forecasting and Chart.js graphs to work, the backend requires historical pricing data for the specific card you are viewing.

-   The data must be in a `.csv` file named exactly after the Card ID (e.g., `xy1-1.csv`).
    
-   The CSV must contain at least two columns: `DATE` (in dd/mm/yyyy format) and `PRICE`.
    
-   Place these CSV files in the same directory as `app.py`. If a CSV is not found, the UI will gracefully display a "No forecast data available" message.
    

## 📄 License

# 

This project utilizes the [TCGdex API](https://tcgdex.dev/), which is free and open-source.
