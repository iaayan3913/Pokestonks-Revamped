# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os
import warnings
import httpx
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can replace "*" with ["http://127.0.0.1:5500"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request Models ====================

class ChatRequest(BaseModel):
    message: str
    model: str = "claude"  # "claude" or "gemini"
    card_context: dict | None = None  # optional card data for context

class ForecastAnalysisRequest(BaseModel):
    card_id: str
    card_name: str
    model: str = "claude"
    historical: list[dict]
    forecast: list[dict]
    current_prices: dict | None = None

class RecommendationRequest(BaseModel):
    card_name: str
    card_type: str | None = None
    card_rarity: str | None = None
    card_set: str | None = None
    current_price: float | None = None
    model: str = "claude"

# ==================== AI Provider Functions ====================

async def call_claude(system_prompt: str, user_message: str) -> str:
    """Call Claude API (Anthropic Messages API)."""
    if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your-claude-api-key-here":
        raise HTTPException(status_code=500, detail="Claude API key not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
        )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Claude API error: {response.text}")
    data = response.json()
    return data["content"][0]["text"]


async def call_gemini(system_prompt: str, user_message: str) -> str:
    """Call Gemini API (Google Generative Language API)."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"content-type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_message}]}],
            },
        )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Gemini API error: {response.text}")
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_ai(model: str, system_prompt: str, user_message: str) -> str:
    """Route to the correct AI provider."""
    if model == "gemini":
        return await call_gemini(system_prompt, user_message)
    else:
        return await call_claude(system_prompt, user_message)


# ==================== AI Status Endpoint ====================

@app.get("/ai/status")
def ai_status():
    """Check which AI models are configured."""
    return {
        "claude": bool(CLAUDE_API_KEY and CLAUDE_API_KEY != "your-claude-api-key-here"),
        "gemini": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here"),
    }


# ==================== AI Chat Endpoint ====================

CHAT_SYSTEM_PROMPT = """You are PokéTracker AI, an expert assistant for Pokémon Trading Card Game collectors and investors.
You have deep knowledge of:
- Pokémon TCG card values, rarities, and market trends
- Card grading (PSA, BGS, CGC)
- Set release history and card availability
- Investment strategies for Pokémon cards
- ARIMA and time-series forecasting concepts (as they apply to card prices)

Keep responses concise, informative, and friendly. Use bullet points when listing information.
If the user provides card context, reference it in your answer.
Format your responses in clean markdown."""

@app.post("/ai/chat")
async def ai_chat(req: ChatRequest):
    user_msg = req.message
    if req.card_context:
        card = req.card_context
        user_msg += f"\n\n[Currently viewing card: {card.get('name', 'Unknown')} (ID: {card.get('id', 'N/A')}), " \
                     f"Set: {card.get('set', 'N/A')}, Rarity: {card.get('rarity', 'N/A')}, " \
                     f"Prices: {card.get('prices', 'N/A')}]"

    reply = await call_ai(req.model, CHAT_SYSTEM_PROMPT, user_msg)
    return {"reply": reply, "model": req.model}


# ==================== AI Forecast Analysis Endpoint ====================

FORECAST_SYSTEM_PROMPT = """You are a Pokémon TCG market analyst AI. You're given historical price data and ARIMA forecast data for a specific card.
Provide a clear, concise analysis covering:
1. **Price Trend**: Is the card trending up, down, or stable? By how much?
2. **Forecast Summary**: What does the model predict? Summarise the predicted direction and magnitude.
3. **Confidence Assessment**: How wide are the confidence intervals? What does that say about reliability?
4. **Investment Take**: Is this a good buy, hold, or sell? Give reasoning.

Keep it under 200 words. Use markdown formatting. Be specific with numbers."""

@app.post("/ai/forecast-analysis")
async def ai_forecast_analysis(req: ForecastAnalysisRequest):
    # Build a data summary for the AI
    hist_prices = [h["price"] for h in req.historical]
    fc_prices = [f["price"] for f in req.forecast]

    recent_price = hist_prices[-1] if hist_prices else 0
    avg_price = sum(hist_prices) / len(hist_prices) if hist_prices else 0
    min_price = min(hist_prices) if hist_prices else 0
    max_price = max(hist_prices) if hist_prices else 0
    final_forecast = fc_prices[-1] if fc_prices else 0
    pct_change = ((final_forecast - recent_price) / recent_price * 100) if recent_price else 0

    user_msg = f"""Card: {req.card_name} (ID: {req.card_id})

Historical Data ({len(req.historical)} data points):
- Current price: ${recent_price:.2f}
- Average: ${avg_price:.2f}, Min: ${min_price:.2f}, Max: ${max_price:.2f}
- Date range: {req.historical[0]['date']} to {req.historical[-1]['date']}

Forecast ({len(req.forecast)} periods ahead):
- Predicted final price: ${final_forecast:.2f} ({pct_change:+.1f}% change)
- Forecast range: ${fc_prices[0]:.2f} to ${fc_prices[-1]:.2f}
- Confidence interval at end: ${req.forecast[-1].get('lower', 0):.2f} - ${req.forecast[-1].get('upper', 0):.2f}

Current Market Prices: {req.current_prices or 'N/A'}"""

    reply = await call_ai(req.model, FORECAST_SYSTEM_PROMPT, user_msg)
    return {"analysis": reply, "model": req.model}


# ==================== AI Card Recommendations Endpoint ====================

RECOMMENDATION_SYSTEM_PROMPT = """You are a Pokémon TCG investment advisor AI. Based on the card the user is viewing, suggest:
1. **Similar Cards**: 3-5 cards with similar characteristics that might be worth collecting
2. **Investment Tips**: Specific advice for this type/rarity of card
3. **Market Context**: Any relevant market trends for this card category

Keep it under 250 words. Use markdown formatting. Be specific with card names and sets when possible."""

@app.post("/ai/recommendations")
async def ai_recommendations(req: RecommendationRequest):
    price_str = f"${req.current_price:.2f}" if req.current_price else "Unknown"
    user_msg = f"""I'm looking at this Pokémon card:
- Name: {req.card_name}
- Type: {req.card_type or 'Unknown'}
- Rarity: {req.card_rarity or 'Unknown'}
- Set: {req.card_set or 'Unknown'}
- Current Price: {price_str}

What similar cards would you recommend, and what's your investment advice?"""

    reply = await call_ai(req.model, RECOMMENDATION_SYSTEM_PROMPT, user_msg)
    return {"recommendations": reply, "model": req.model}


# ==================== Existing Endpoints ====================

@app.get("/has_data/{card_id}")
def has_data(card_id: str):
    filename = f"{card_id}.csv"
    return {"exists": os.path.exists(filename)}


def find_best_arima(data, max_p=5, max_d=2, max_q=5):
    """Grid-search (p,d,q) for the lowest AIC, suppressing noisy warnings."""
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_model = fitted
                except Exception:
                    continue
    return best_model, best_order, best_aic


@app.get("/forecast/{card_id}")
def forecast(card_id: str, steps: int = 6):

    csv_path = f"{card_id}.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV not found for this card")

    # Load and prepare data
    df = pd.read_csv(csv_path)
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
    df = df.sort_values('DATE').drop_duplicates(subset=['DATE'], keep='first')

    # Set a proper weekly frequency so statsmodels doesn't guess
    df.set_index('DATE', inplace=True)
    df = df.asfreq('W-MON')                          # explicit weekly frequency
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
    df['PRICE'] = df['PRICE'].interpolate(method='linear').ffill().bfill()

    # Find best ARIMA (warnings silenced inside the function)
    best_model, best_order, best_aic = find_best_arima(df['PRICE'])

    if best_model is None:
        raise HTTPException(status_code=500, detail="Could not fit any ARIMA model to this data")

    # Forecast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_result = best_model.get_forecast(steps=steps)
    fc = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                   periods=steps, freq='W-MON')

    # Build JSON response
    return {
        "card_id": card_id,
        "best_order": best_order,
        "aic": best_aic,
        "historical": [
            {"date": str(d.date()), "price": float(p)}
            for d, p in zip(df.index, df['PRICE'])
        ],
        "forecast": [
            {
                "date": str(d.date()),
                "price": float(p),
                "lower": float(conf_int.iloc[i, 0]),
                "upper": float(conf_int.iloc[i, 1])
            }
            for i, (d, p) in enumerate(zip(forecast_dates, fc))
        ]
    }
