from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import io
import requests
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import os
import hashlib
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Import translations
from app.translations import TRANSLATIONS, get_translation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional models with auto-activation
try:
    import pmdarima as pm
    HAS_ARIMA = True
    logger.info("✅ ARIMA model activated (pmdarima installed)")
except ImportError:
    HAS_ARIMA = False
    logger.warning("⚠️ ARIMA model not available (pmdarima not installed)")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
    logger.info("✅ XGBoost model activated (xgboost installed)")
except ImportError:
    HAS_XGB = False
    logger.warning("⚠️ XGBoost model not available (xgboost not installed)")

app = FastAPI(title="Global SDGs Analytics Platform")

# Enhanced cache system with TTL
CACHE_DIR = os.path.join("app", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = 3600 * 24  # 24 hours

# Rate limiting configuration
RATE_LIMITS = {
    "world_bank": {"calls": 60, "period": 60},  # 60 calls per minute
    "oecd": {"calls": 30, "period": 60},
    "owid": {"calls": 30, "period": 60}
}

class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove old calls outside the period
        self.call_times = [t for t in self.call_times if now - t < self.period]
        
        if len(self.call_times) >= self.calls:
            # Need to wait
            sleep_time = self.period - (now - self.call_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.call_times.append(now)

# Initialize rate limiters
rate_limiters = {
    source: RateLimiter(**config) 
    for source, config in RATE_LIMITS.items()
}

# Session with retry and backoff
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = get_session()

# Enhanced caching with TTL
def cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key from prefix and parameters."""
    key_str = f"{prefix}::" + "::".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.sha256(key_str.encode()).hexdigest()

def save_cache(key: str, data: pd.DataFrame):
    """Save DataFrame to cache with timestamp."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta.json")
    
    try:
        data.to_parquet(cache_path)
        with open(meta_path, 'w') as f:
            json.dump({"timestamp": time.time()}, f)
    except Exception as e:
        logger.error(f"Cache save error: {e}")

def load_cache(key: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from cache if not expired."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.parquet")
    meta_path = os.path.join(CACHE_DIR, f"{key}.meta.json")
    
    try:
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            if time.time() - meta["timestamp"] < CACHE_TTL:
                return pd.read_parquet(cache_path)
    except Exception as e:
        logger.error(f"Cache load error: {e}")
    
    return None

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Extended indicator registry with more SDGs
INDICATORS = {
    # SDG 1: No Poverty
    "SI.POV.DDAY": {"name": "Poverty headcount ratio at $2.15 a day", "sdg": "SDG 1", "unit": "%", "target": 0.0, "direction": "<="},
    "SI.POV.NAHC": {"name": "Poverty headcount ratio at national poverty lines", "sdg": "SDG 1", "unit": "%", "target": 0.0, "direction": "<="},
    
    # SDG 2: Zero Hunger
    "SN.ITK.DEFC.ZS": {"name": "Prevalence of undernourishment", "sdg": "SDG 2", "unit": "%", "target": 0.0, "direction": "<="},
    
    # SDG 3: Good Health
    "SH.DYN.MORT": {"name": "Under-5 mortality rate", "sdg": "SDG 3", "unit": "per 1,000", "target": 25.0, "direction": "<="},
    "SH.STA.MMRT": {"name": "Maternal mortality ratio", "sdg": "SDG 3", "unit": "per 100,000", "target": 70.0, "direction": "<="},
    
    # SDG 4: Quality Education  
    "SE.PRM.NENR": {"name": "Primary school enrollment", "sdg": "SDG 4", "unit": "%", "target": 100.0, "direction": ">="},
    "SE.SEC.NENR": {"name": "Secondary school enrollment", "sdg": "SDG 4", "unit": "%", "target": 100.0, "direction": ">="},
    "SE.ADT.LITR.ZS": {"name": "Adult literacy rate", "sdg": "SDG 4", "unit": "%", "target": 100.0, "direction": ">="},
    
    # SDG 5: Gender Equality
    "SG.GEN.PARL.ZS": {"name": "Women in parliament", "sdg": "SDG 5", "unit": "%", "target": 50.0, "direction": ">="},
    
    # SDG 6: Clean Water
    "SH.H2O.BASW.ZS": {"name": "Access to basic drinking water", "sdg": "SDG 6", "unit": "%", "target": 100.0, "direction": ">="},
    
    # SDG 7: Clean Energy
    "EG.ELC.ACCS.ZS": {"name": "Access to electricity", "sdg": "SDG 7", "unit": "%", "target": 100.0, "direction": ">="},
    "EG.FEC.RNEW.ZS": {"name": "Renewable energy consumption", "sdg": "SDG 7", "unit": "%", "target": 50.0, "direction": ">="},
    
    # SDG 8: Economic Growth
    "NY.GDP.PCAP.KD.ZG": {"name": "GDP per capita growth", "sdg": "SDG 8", "unit": "%", "target": 3.0, "direction": ">="},
    "SL.UEM.TOTL.ZS": {"name": "Unemployment rate", "sdg": "SDG 8", "unit": "%", "target": 5.0, "direction": "<="},
    
    # SDG 13: Climate Action
    "EN.ATM.CO2E.PC": {"name": "CO2 emissions per capita", "sdg": "SDG 13", "unit": "tons", "target": 2.0, "direction": "<="},
    "EN.ATM.PM25.MC.M3": {"name": "PM2.5 air pollution", "sdg": "SDG 13", "unit": "µg/m³", "target": 10.0, "direction": "<="},
}

# Country list (subset for demo)
COUNTRIES = {
    "KR": "Korea, Rep.",
    "US": "United States", 
    "CN": "China",
    "JP": "Japan",
    "DE": "Germany",
    "FR": "France",
    "GB": "United Kingdom",
    "IN": "India",
    "BR": "Brazil",
    "ZA": "South Africa",
    "NG": "Nigeria",
    "EG": "Egypt, Arab Rep.",
    "AU": "Australia",
    "CA": "Canada",
    "MX": "Mexico",
    "ID": "Indonesia",
    "TR": "Turkey",
    "SA": "Saudi Arabia",
    "AR": "Argentina",
    "RU": "Russian Federation"
}

# Data fetching functions with rate limiting and caching
async def fetch_world_bank(country: str, indicator: str) -> pd.DataFrame:
    """Fetch data from World Bank API with caching and rate limiting."""
    cache_id = cache_key("wb", country=country, indicator=indicator)
    cached = load_cache(cache_id)
    if cached is not None:
        return cached
    
    rate_limiters["world_bank"].wait_if_needed()
    
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 1000}
    
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            raise ValueError("No data available")
        
        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "year": int(item["date"]),
                    "value": float(item["value"])
                })
        
        if not records:
            raise ValueError("No valid data points")
        
        df = pd.DataFrame(records).sort_values("year")
        save_cache(cache_id, df)
        return df
        
    except Exception as e:
        logger.error(f"World Bank fetch error: {e}")
        raise

async def fetch_owid(indicator: str, country: str) -> pd.DataFrame:
    """Fetch data from Our World in Data."""
    cache_id = cache_key("owid", indicator=indicator, country=country)
    cached = load_cache(cache_id)
    if cached is not None:
        return cached
    
    rate_limiters["owid"].wait_if_needed()
    
    url = f"https://github.com/owid/owid-datasets/raw/master/datasets/{indicator}/{indicator}.csv"
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        df = df[df["Entity"] == country][["Year", "Value"]]
        df.columns = ["year", "value"]
        df = df.dropna().sort_values("year")
        
        if df.empty:
            raise ValueError("No data for country")
        
        save_cache(cache_id, df)
        return df
        
    except Exception as e:
        logger.error(f"OWID fetch error: {e}")
        raise

async def fetch_oecd(dataset: str, country: str) -> pd.DataFrame:
    """Fetch data from OECD API."""
    cache_id = cache_key("oecd", dataset=dataset, country=country)
    cached = load_cache(cache_id)
    if cached is not None:
        return cached
    
    rate_limiters["oecd"].wait_if_needed()
    
    url = f"https://stats.oecd.org/SDMX-JSON/data/{dataset}/{country}"
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        # Parse OECD JSON structure (simplified)
        # Implementation would need proper SDMX-JSON parsing
        
        df = pd.DataFrame()  # Placeholder
        save_cache(cache_id, df)
        return df
        
    except Exception as e:
        logger.error(f"OECD fetch error: {e}")
        raise

# Batch ETL function
async def batch_fetch_indicators(country: str, indicators: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch multiple indicators in parallel with proper error handling."""
    results = {}
    tasks = []
    
    for indicator in indicators:
        task = fetch_world_bank(country, indicator)
        tasks.append((indicator, task))
    
    for indicator, task in tasks:
        try:
            df = await task
            results[indicator] = df
        except Exception as e:
            logger.error(f"Failed to fetch {indicator}: {e}")
            results[indicator] = None
    
    return results

# Enhanced forecasting with confidence intervals
def forecast_with_linear(data: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """Linear regression with prediction intervals."""
    X = data[["year"]].values
    y = data["value"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate residual standard error
    predictions = model.predict(X)
    residuals = y - predictions
    rse = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
    
    # Future predictions
    last_year = data["year"].max()
    future_years = np.arange(last_year + 1, last_year + horizon + 1).reshape(-1, 1)
    future_pred = model.predict(future_years)
    
    # 95% confidence interval
    t_stat = 1.96  # approximation for large samples
    se = rse * np.sqrt(1 + 1/len(y) + (future_years - X.mean())**2 / np.sum((X - X.mean())**2))
    lower = future_pred - t_stat * se.flatten()
    upper = future_pred + t_stat * se.flatten()
    
    # Calculate R²
    r2 = r2_score(y, predictions)
    
    return {
        "years": future_years.flatten().tolist(),
        "predictions": future_pred.tolist(),
        "lower_bound": lower.tolist(),
        "upper_bound": upper.tolist(),
        "r2": r2,
        "model": "Linear Regression"
    }

def forecast_with_arima(data: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """ARIMA forecasting with confidence intervals."""
    if not HAS_ARIMA:
        raise ValueError("ARIMA not available")
    
    values = data["value"].values
    
    # Auto ARIMA
    model = pm.auto_arima(values, seasonal=False, stepwise=True, suppress_warnings=True)
    
    # Forecast
    forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
    
    last_year = data["year"].max()
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    
    return {
        "years": future_years,
        "predictions": forecast.tolist(),
        "lower_bound": conf_int[:, 0].tolist(),
        "upper_bound": conf_int[:, 1].tolist(),
        "model": "ARIMA"
    }

def forecast_with_xgboost(data: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """XGBoost forecasting with confidence intervals."""
    if not HAS_XGB:
        raise ValueError("XGBoost not available")
    
    # Feature engineering
    data = data.copy()
    data["year_normalized"] = (data["year"] - data["year"].min()) / (data["year"].max() - data["year"].min())
    data["trend"] = np.arange(len(data))
    
    X = data[["year_normalized", "trend"]].values
    y = data["value"].values
    
    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Predictions on training data for residuals
    train_pred = model.predict(X)
    residuals = y - train_pred
    rse = np.std(residuals)
    
    # Future predictions
    last_year = data["year"].max()
    future_years = np.arange(last_year + 1, last_year + horizon + 1)
    
    year_min = data["year"].min()
    year_range = data["year"].max() - year_min
    future_normalized = (future_years - year_min) / year_range
    future_trend = np.arange(len(data), len(data) + horizon)
    
    X_future = np.column_stack([future_normalized, future_trend])
    future_pred = model.predict(X_future)
    
    # Confidence intervals (approximate)
    t_stat = 1.96
    lower = future_pred - t_stat * rse
    upper = future_pred + t_stat * rse
    
    # R² score
    r2 = r2_score(y, train_pred)
    
    return {
        "years": future_years.tolist(),
        "predictions": future_pred.tolist(),
        "lower_bound": lower.tolist(),
        "upper_bound": upper.tolist(),
        "r2": r2,
        "model": "XGBoost"
    }

# Calculate achievement percentage
def calculate_achievement(current: float, target: float, direction: str, initial: float = None) -> float:
    """Calculate achievement percentage towards SDG target."""
    if direction == ">=":
        # Higher is better
        if target == 0:
            return 100.0 if current > 0 else 0.0
        return min(100.0, (current / target) * 100)
    else:
        # Lower is better
        if initial is None:
            initial = current * 2  # Rough estimate
        
        if target >= initial:
            return 100.0
        
        progress = (initial - current) / (initial - target)
        return max(0.0, min(100.0, progress * 100))

# API Endpoints
@app.get("/")
async def home(request: Request, lang: str = Query(default="en")):
    """Render home page with selected language."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "lang": lang,
        "translations": TRANSLATIONS[lang] if lang in TRANSLATIONS else TRANSLATIONS["en"],
        "countries": COUNTRIES,
        "indicators": INDICATORS,
        "models": {
            "linear": "Linear Regression",
            "arima": "ARIMA (Auto)" if HAS_ARIMA else "ARIMA (Not Available)",
            "xgboost": "XGBoost" if HAS_XGB else "XGBoost (Not Available)"
        },
        "languages": {
            "en": "English",
            "ko": "한국어",
            "fr": "Français",
            "zh": "中文",
            "ja": "日本語"
        }
    })

@app.get("/api/capabilities")
async def get_capabilities():
    """Get system capabilities."""
    return {
        "models": {
            "linear": True,
            "arima": HAS_ARIMA,
            "xgboost": HAS_XGB
        },
        "data_sources": ["world_bank", "owid", "oecd"],
        "cache_ttl": CACHE_TTL,
        "rate_limits": RATE_LIMITS
    }

@app.post("/api/analyze")
async def analyze(
    country: str = Form(...),
    indicators: List[str] = Form(...),
    model: str = Form("linear"),
    horizon: int = Form(5),
    lang: str = Form("en")
):
    """Analyze multiple indicators with forecasting."""
    try:
        # Fetch all indicators
        data_dict = await batch_fetch_indicators(country, indicators)
        
        results = []
        for indicator, data in data_dict.items():
            if data is None or data.empty:
                results.append({
                    "indicator": indicator,
                    "error": "No data available"
                })
                continue
            
            # Get indicator metadata
            meta = INDICATORS.get(indicator, {})
            
            # Perform forecasting
            try:
                if model == "linear":
                    forecast = forecast_with_linear(data, horizon)
                elif model == "arima":
                    forecast = forecast_with_arima(data, horizon)
                elif model == "xgboost":
                    forecast = forecast_with_xgboost(data, horizon)
                else:
                    raise ValueError(f"Unknown model: {model}")
            except Exception as e:
                results.append({
                    "indicator": indicator,
                    "error": str(e)
                })
                continue
            
            # Calculate achievement
            current_value = data["value"].iloc[-1]
            predicted_value = forecast["predictions"][-1]
            initial_value = data["value"].iloc[0] if len(data) > 0 else None
            
            achievement_current = calculate_achievement(
                current_value,
                meta.get("target", 0),
                meta.get("direction", ">="),
                initial_value
            )
            
            achievement_predicted = calculate_achievement(
                predicted_value,
                meta.get("target", 0),
                meta.get("direction", ">="),
                initial_value
            )
            
            results.append({
                "indicator": indicator,
                "name": meta.get("name", indicator),
                "sdg": meta.get("sdg", ""),
                "unit": meta.get("unit", ""),
                "target": meta.get("target", None),
                "historical": {
                    "years": data["year"].tolist(),
                    "values": data["value"].tolist()
                },
                "forecast": forecast,
                "achievement": {
                    "current": achievement_current,
                    "predicted": achievement_predicted,
                    "target": meta.get("target", None),
                    "direction": meta.get("direction", ">=")
                }
            })
        
        return {
            "success": True,
            "country": COUNTRIES.get(country, country),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/upload")
async def upload_custom_data(
    file: UploadFile = File(...),
    model: str = Form("linear"),
    horizon: int = Form(5)
):
    """Upload and analyze custom CSV data."""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate columns
        if "year" not in df.columns or "value" not in df.columns:
            raise ValueError("CSV must contain 'year' and 'value' columns")
        
        df = df[["year", "value"]].dropna().sort_values("year")
        
        # Perform forecasting
        if model == "linear":
            forecast = forecast_with_linear(df, horizon)
        elif model == "arima":
            forecast = forecast_with_arima(df, horizon)
        elif model == "xgboost":
            forecast = forecast_with_xgboost(df, horizon)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        return {
            "success": True,
            "historical": {
                "years": df["year"].tolist(),
                "values": df["value"].tolist()
            },
            "forecast": forecast
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/export/{format}")
async def export_results(
    format: str,
    country: str = Query(...),
    indicators: str = Query(...),
    model: str = Query("linear"),
    horizon: int = Query(5)
):
    """Export analysis results in various formats."""
    # Implementation for CSV/JSON/Excel export
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)