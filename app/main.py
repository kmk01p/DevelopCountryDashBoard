from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io
import requests
from typing import Dict, Any, List, Optional

app = FastAPI(title="SDGs AI Dashboard")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Supported indicators (World Bank codes) mapped to friendly names and SDG
SUPPORTED_INDICATORS: Dict[str, Dict[str, str]] = {
    # Education (SDG 4)
    "SE.PRM.NENR": {"name": "Primary school net enrollment rate (% of primary school age children)", "sdg": "SDG 4"},
    "SE.SEC.NENR": {"name": "Secondary school net enrollment rate (% of secondary school age population)", "sdg": "SDG 4"},
    "SE.ADT.LITR.ZS": {"name": "Adult literacy rate, population 15+ years, both sexes (%)", "sdg": "SDG 4"},
    # Poverty (SDG 1)
    "SI.POV.DDAY": {"name": "Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)", "sdg": "SDG 1"},
    # Climate (SDG 13)
    "EN.ATM.CO2E.PC": {"name": "CO2 emissions (metric tons per capita)", "sdg": "SDG 13"},
    "EN.ATM.CO2E.KT": {"name": "CO2 emissions (kt)", "sdg": "SDG 13"},
    "EN.ATM.PM25.MC.M3": {"name": "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)", "sdg": "SDG 3/13"},
    # Health (SDG 3)
    "SH.DYN.MORT": {"name": "Mortality rate, under-5 (per 1,000 live births)", "sdg": "SDG 3"},
}


def _train_and_forecast(df: pd.DataFrame, horizon: int, target_column: str = "value") -> Dict[str, Any]:
    # basic validation
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not in data. Columns: {list(df.columns)}"}
    if "year" not in df.columns:
        return {"error": "Data must include a 'year' column."}

    df = df.dropna(subset=["year", target_column]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")

    if len(df) < 3:
        return {"error": "Not enough data points to train a model (need >= 3)."}

    X = df[["year"]]
    y = df[target_column]

    # Use time-ordered split (no shuffle)
    test_size = max(1, int(len(df) * 0.2))
    split_idx = len(df) - test_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = None
    if len(y_test) > 1:
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

    last_year = int(df["year"].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    future_df = pd.DataFrame({"year": future_years})
    future_preds = model.predict(future_df[["year"]]).tolist()

    return {
        "historical": {
            "year": df["year"].tolist(),
            target_column: df[target_column].tolist(),
        },
        "metrics": {"r2": score},
        "forecast": {"year": future_years, "pred": future_preds},
    }


def fetch_world_bank_series(country: str, indicator: str, per_page: int = 20000) -> pd.DataFrame:
    """Fetch time series data from World Bank API for a country and indicator.
    Returns a DataFrame with columns: year, value
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page={per_page}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"World Bank API request failed with status {resp.status_code}")
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError("Unexpected World Bank API response format")
    series = data[1] or []
    rows: List[Dict[str, Any]] = []
    for item in series:
        year = item.get("date")
        value = item.get("value")
        if year is None:
            continue
        try:
            y = int(year)
        except Exception:
            continue
        # value can be None
        rows.append({"year": y, "value": value})
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["value"]).copy()
    if df.empty:
        raise RuntimeError("No valid data points returned from World Bank.")
    df = df.sort_values("year")
    return df


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/indicators")
async def list_indicators():
    return {"indicators": [{"code": k, **v} for k, v in SUPPORTED_INDICATORS.items()]}


@app.get("/api/fetch_wb")
async def api_fetch_wb(country: str, indicator: str):
    """Fetch raw World Bank series for preview (no forecast)."""
    try:
        df = fetch_world_bank_series(country=country, indicator=indicator)
        return {"year": df["year"].tolist(), "value": df["value"].tolist()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_wb")
async def predict_wb(country: str, indicator: str, horizon: int = 5):
    """Fetch World Bank data and produce forecast."""
    try:
        df = fetch_world_bank_series(country=country, indicator=indicator)
        result = _train_and_forecast(df, horizon=horizon, target_column="value")
        # Attach meta
        result["meta"] = {
            "country": country,
            "indicator": indicator,
            "indicator_name": SUPPORTED_INDICATORS.get(indicator, {}).get("name"),
            "sdg": SUPPORTED_INDICATORS.get(indicator, {}).get("sdg"),
        }
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  horizon: int = Form(5),
                  target_column: str = Form("value")):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    result = _train_and_forecast(df, horizon=horizon, target_column=target_column)
    return result
