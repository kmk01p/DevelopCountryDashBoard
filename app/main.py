from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io
import requests
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache

# Optional models
try:
    import pmdarima as pm  # ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

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

# Simple goal registry for achievement scoring
GOAL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # 100% enrollment and literacy desired
    "SE.PRM.NENR": {"target": 100.0, "direction": ">=", "unit": "%"},
    "SE.SEC.NENR": {"target": 100.0, "direction": ">=", "unit": "%"},
    "SE.ADT.LITR.ZS": {"target": 100.0, "direction": ">=", "unit": "%"},
    # Poverty target is 0%
    "SI.POV.DDAY": {"target": 0.0, "direction": "<=", "unit": "%"},
    # Emissions ideally reduced (no global fixed target here) – we measure reduction vs first year
    "EN.ATM.CO2E.PC": {"target": 0.0, "direction": "<=", "unit": "t/capita"},
    "EN.ATM.CO2E.KT": {"target": 0.0, "direction": "<=", "unit": "kt"},
    "EN.ATM.PM25.MC.M3": {"target": 5.0, "direction": "<=", "unit": "µg/m³"},  # WHO guideline ~5
    "SH.DYN.MORT": {"target": 25.0, "direction": "<=", "unit": "per 1,000"},   # SDG 3.2 target example
}


def _time_split(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    test_size = max(1, int(len(df) * 0.2))
    split_idx = len(df) - test_size
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def _linear_model(X_train, y_train, X_test, y_test, X_future) -> Dict[str, Any]:
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) if len(X_test) else []
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else None
    # naive PI via residual std
    resid_std = None
    try:
        in_pred = model.predict(X_train)
        resid_std = float(pd.Series(y_train - in_pred).std())
    except Exception:
        pass
    fut_pred = model.predict(X_future)
    if resid_std is not None:
        ci = 1.96 * resid_std
        lower = (fut_pred - ci).tolist()
        upper = (fut_pred + ci).tolist()
    else:
        lower = [None] * len(fut_pred)
        upper = [None] * len(fut_pred)
    return {"pred": fut_pred.tolist(), "r2": r2, "pi_low": lower, "pi_high": upper, "model": "linear"}


def _arima_model(series: List[float], horizon: int) -> Dict[str, Any]:
    if not HAS_ARIMA:
        return {"error": "ARIMA not available on server", "model": "arima"}
    try:
        model = pm.auto_arima(series, seasonal=False, error_action='ignore', suppress_warnings=True)
        fc, conf = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        return {"pred": fc.tolist(), "pi_low": conf[:, 0].tolist(), "pi_high": conf[:, 1].tolist(), "model": "arima"}
    except Exception as e:
        return {"error": str(e), "model": "arima"}


def _xgb_model(X_train, y_train, X_test, y_test, X_future) -> Dict[str, Any]:
    if not HAS_XGB:
        return {"error": "XGBoost not available on server", "model": "xgboost"}
    try:
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test) if len(X_test) else []
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else None
        # naive PI via residual std on train
        resid_std = None
        try:
            in_pred = model.predict(X_train)
            resid_std = float(pd.Series(y_train - in_pred).std())
        except Exception:
            pass
        fut_pred = model.predict(X_future)
        if resid_std is not None:
            ci = 1.96 * resid_std
            lower = (fut_pred - ci).tolist()
            upper = (fut_pred + ci).tolist()
        else:
            lower = [None] * len(fut_pred)
            upper = [None] * len(fut_pred)
        return {"pred": fut_pred.tolist(), "r2": r2, "pi_low": lower, "pi_high": upper, "model": "xgboost"}
    except Exception as e:
        return {"error": str(e), "model": "xgboost"}


def _achievement(latest_val: float, forecast_last: float, indicator: Optional[str], first_hist: Optional[float] = None) -> Dict[str, Any]:
    meta = GOAL_REGISTRY.get(indicator or "", None)
    if not meta:
        return {"defined": False}
    direction = meta["direction"]
    target = float(meta["target"])
    # progress function returns percent [0,100+]
    def prog(current: float) -> Optional[float]:
        if direction == ">=":
            if target == 0:
                return None
            return max(0.0, min(100.0, 100.0 * current / target))
        # <= direction
        baseline = first_hist if first_hist is not None else None
        if target == 0.0:
            if baseline in (None, 0):
                return None
            # 100% when current==0, 0% when current==baseline
            return max(0.0, min(100.0, 100.0 * (baseline - current) / baseline))
        else:
            # target > 0
            if baseline is None:
                return None
            total_gap = max(1e-9, baseline - target)
            return max(0.0, min(100.0, 100.0 * (baseline - current) / total_gap))
    return {
        "defined": True,
        "target": target,
        "direction": direction,
        "latest_progress_pct": prog(latest_val),
        "forecast_progress_pct": prog(forecast_last),
        "unit": meta.get("unit"),
    }


def _prepare_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    # keep numeric columns, fill forward then drop remaining NaNs
    num_df = df.select_dtypes(include=["number"]).copy()
    num_df = num_df.sort_values("year")
    num_df = num_df.ffill().bfill()
    # ensure target exists
    if target_column not in num_df.columns:
        raise ValueError(f"Target column '{target_column}' not numeric or not present.")
    return num_df


def _train_and_forecast(df: pd.DataFrame, horizon: int, target_column: str = "value", model: str = "linear", indicator: Optional[str] = None) -> Dict[str, Any]:
    if "year" not in df.columns:
        return {"error": "Data must include a 'year' column."}
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year")
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not in data. Columns: {list(df.columns)}"}
    if len(df) < 3:
        return {"error": "Not enough data points to train a model (need >= 3)."}

    num_df = _prepare_features(df, target_column)
    # If only 1 feature (target), add year as feature to allow trend
    features = [c for c in num_df.columns if c not in (target_column, "year")]
    if features == []:
        # ensure year exists as a feature
        if "year" not in num_df.columns:
            num_df = pd.concat([num_df, df[["year"]]], axis=1)
        features = ["year"]
    # Build train/test split with time order (avoid duplicate year column)
    cols = ["year"] + features + [target_column]
    # remove duplicates while preserving order
    seen = set()
    cols = [x for x in cols if not (x in seen or seen.add(x))]
    full = num_df[cols].drop_duplicates("year").sort_values("year")
    if len(full) < 3:
        return {"error": "Insufficient unique yearly data."}

    X = full[features]
    y = full[target_column]
    test_size = max(1, int(len(full) * 0.2))
    split_idx = len(full) - test_size
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    last_year = int(full["year"].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    # For features that are not 'year', keep them constant by last known value (simple baseline)
    future_feat = {}
    for f in features:
        if f == "year":
            future_feat[f] = future_years
        else:
            future_feat[f] = [float(full[f].iloc[-1])] * len(future_years)
    X_future = pd.DataFrame(future_feat)

    result: Dict[str, Any]
    if model == "linear":
        out = _linear_model(X_train, y_train, X_test, y_test, X_future)
        result = out
    elif model == "arima":
        # use target series only, ignore features
        out = _arima_model(full[target_column].tolist(), horizon)
        if "error" in out:
            return {"error": out["error"], "model": "arima"}
        result = {"pred": out["pred"], "r2": None, "pi_low": out["pi_low"], "pi_high": out["pi_high"], "model": "arima"}
    elif model == "xgboost":
        out = _xgb_model(X_train, y_train, X_test, y_test, X_future)
        if "error" in out:
            return out
        result = out
    else:
        return {"error": f"Unknown model '{model}'", "model": model}

    hist_years = full["year"].tolist()
    hist_vals = full[target_column].tolist()

    achievement = _achievement(latest_val=hist_vals[-1], forecast_last=float(result["pred"][-1]), indicator=indicator, first_hist=hist_vals[0])

    return {
        "historical": {"year": hist_years, target_column: hist_vals},
        "metrics": {"r2": result.get("r2"), "model": result.get("model")},
        "forecast": {"year": future_years, "pred": result["pred"], "pi_low": result.get("pi_low"), "pi_high": result.get("pi_high")},
        "achievement": achievement,
    }


@lru_cache(maxsize=256)
def fetch_world_bank_series(country: str, indicator: str, per_page: int = 20000) -> pd.DataFrame:
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
        rows.append({"year": y, indicator: value})
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise RuntimeError("No valid data points returned from World Bank.")
    return df.sort_values("year")


def fetch_world_bank_multi(country: str, indicators: List[str]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for ind in indicators:
        part = fetch_world_bank_series(country=country, indicator=ind)
        merged = part if merged is None else pd.merge(merged, part, on="year", how="outer")
    if merged is None or merged.empty:
        raise RuntimeError("No data merged from World Bank.")
    merged = merged.sort_values("year").dropna(how="all")
    return merged


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/indicators")
async def list_indicators():
    caps = {"arima": HAS_ARIMA, "xgboost": HAS_XGB, "linear": True}
    return {"indicators": [{"code": k, **v} for k, v in SUPPORTED_INDICATORS.items()], "capabilities": caps}


@app.get("/api/fetch_wb")
async def api_fetch_wb(country: str, indicator: str):
    try:
        df = fetch_world_bank_series(country=country, indicator=indicator)
        return {"year": df["year"].tolist(), "value": df[indicator].tolist()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_wb")
async def predict_wb(country: str, indicators: str, horizon: int = 5, model: str = "linear", target_indicator: Optional[str] = None):
    """Fetch one or multiple WB indicators (comma-separated) and forecast target.
    If multiple indicators are provided and target_indicator omitted, the first is used as target; others become features.
    """
    try:
        codes = [c.strip() for c in indicators.split(',') if c.strip()]
        if not codes:
            return {"error": "No indicator codes provided."}
        df = fetch_world_bank_multi(country=country, indicators=codes)
        tgt = target_indicator or codes[0]
        if tgt not in df.columns:
            return {"error": f"Target indicator '{tgt}' not present in merged data."}
        result = _train_and_forecast(df, horizon=horizon, target_column=tgt, model=model, indicator=tgt)
        result["meta"] = {
            "country": country,
            "indicators": codes,
            "target_indicator": tgt,
            "indicator_name": SUPPORTED_INDICATORS.get(tgt, {}).get("name"),
            "sdg": SUPPORTED_INDICATORS.get(tgt, {}).get("sdg"),
        }
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  horizon: int = Form(5),
                  target_column: str = Form("value"),
                  model: str = Form("linear")):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    result = _train_and_forecast(df, horizon=horizon, target_column=target_column, model=model)
    return result
