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
import os
import hashlib
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# Simple rate limiter and retry for external calls
import time
from functools import wraps

LAST_CALL_TS = 0.0
MIN_INTERVAL = 0.5  # seconds

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global LAST_CALL_TS
        wait = MIN_INTERVAL - (time.time() - LAST_CALL_TS)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                result = func(*args, **kwargs)
                LAST_CALL_TS = time.time()
                return result
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))  # backoff
    return wrapper

app = FastAPI(title="SDGs AI Dashboard")

# Simple on-disk cache + rate-limited session for external data sources
CACHE_DIR = os.path.join("app", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_session = None
_last_call_ts = 0.0
RATE_LIMIT_SECONDS = 0.2  # ~5 req/sec

def get_session():
    global _session
    if _session is None:
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
    return _session

def rl_get(url: str, timeout: int = 30):
    global _last_call_ts
    now = time.time()
    wait = _last_call_ts + RATE_LIMIT_SECONDS - now
    if wait > 0:
        time.sleep(wait)
    resp = get_session().get(url, timeout=timeout)
    _last_call_ts = time.time()
    return resp

def _cache_path(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.parquet")

def cache_df(key: str, df: pd.DataFrame):
    try:
        path = _cache_path(key)
        df.to_parquet(path)
    except Exception:
        pass

def load_cache_df(key: str) -> Optional[pd.DataFrame]:
    try:
        path = _cache_path(key)
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

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
@rate_limited
def fetch_world_bank_series(country: str, indicator: str, per_page: int = 20000) -> pd.DataFrame:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page={per_page}"
    ck = f"wb::{country}::{indicator}"
    cached = load_cache_df(ck)
    if cached is not None and not cached.empty:
        return cached
    resp = rl_get(url, timeout=30)
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
    df = df.sort_values("year")
    cache_df(ck, df)
    return df


def fetch_world_bank_multi(country: str, indicators: List[str]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for ind in indicators:
        part = fetch_world_bank_series(country=country, indicator=ind)
        merged = part if merged is None else pd.merge(merged, part, on="year", how="outer")
    if merged is None or merged.empty:
        raise RuntimeError("No data merged from World Bank.")
    merged = merged.sort_values("year").dropna(how="all")
    return merged


def fetch_owid_series(indicator: str, country_code: str) -> pd.DataFrame:
    """Fetch a series from Our World in Data Grapher CSV (columns: Entity, Code, Year, Value)."""
    url = f"https://ourworldindata.org/grapher/{indicator}.csv"
    ck = f"owid::{indicator}::{country_code}"
    cached = load_cache_df(ck)
    if cached is not None and not cached.empty:
        return cached
    resp = rl_get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OWID fetch failed ({resp.status_code})")
    df = pd.read_csv(io.BytesIO(resp.content))
    if not set(["Code","Year","Value"]).issubset(df.columns):
        raise RuntimeError("Unexpected OWID CSV format")
    df = df[df["Code"] == country_code]
    df = df.rename(columns={"Year":"year","Value":"value"})[["year","value"]].dropna()
    df = df.sort_values("year")
    if df.empty:
        raise RuntimeError("No OWID data for specified country/indicator")
    cache_df(ck, df)
    return df


def fetch_oecd_series(dataset: str, key: str) -> pd.DataFrame:
    """Fetch a series from OECD SDMX-JSON CSV output; expects TIME_PERIOD and Value columns."""
    url = f"https://stats.oecd.org/sdmx-json/data/{dataset}/{key}?contentType=csv"
    ck = f"oecd::{dataset}::{key}"
    cached = load_cache_df(ck)
    if cached is not None and not cached.empty:
        return cached
    resp = rl_get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OECD fetch failed ({resp.status_code})")
    df = pd.read_csv(io.BytesIO(resp.content))
    if not set(["TIME_PERIOD","Value"]).issubset(df.columns):
        raise RuntimeError("Unexpected OECD CSV format; please adjust dataset/key")
    def parse_year(x):
        try:
            return int(str(x)[:4])
        except Exception:
            return None
    df["year"] = df["TIME_PERIOD"].apply(parse_year)
    df = df.dropna(subset=["year","Value"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.rename(columns={"Value":"value"})[["year","value"]].sort_values("year")
    if df.empty:
        raise RuntimeError("No OECD data after parsing")
    cache_df(ck, df)
    return df


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


@app.get("/predict_owid")
async def predict_owid(indicator: str, country: str, horizon: int = 5, model: str = "linear"):
    try:
        df = fetch_owid_series(indicator=indicator, country_code=country)
        result = _train_and_forecast(df, horizon=horizon, target_column="value", model=model, indicator=indicator)
        result["meta"] = {"source": "OWID", "indicator": indicator, "country": country}
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/predict_oecd")
async def predict_oecd(dataset: str, key: str, horizon: int = 5, model: str = "linear"):
    try:
        df = fetch_oecd_series(dataset=dataset, key=key)
        result = _train_and_forecast(df, horizon=horizon, target_column="value", model=model, indicator=dataset)
        result["meta"] = {"source": "OECD", "dataset": dataset, "key": key}
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
