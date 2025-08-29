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

app = FastAPI(title="SDGs AI Dashboard")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  horizon: int = Form(5),
                  target_column: str = Form("value")):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not in CSV. Columns: {list(df.columns)}"}
    # simple time-based features assuming a year column exists
    if 'year' not in df.columns:
        return {"error": "CSV must include a 'year' column."}

    df = df.sort_values('year')
    X = df[['year']]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred) if len(y_test) > 1 else None

    last_year = int(df['year'].max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    future_df = pd.DataFrame({'year': future_years})
    future_preds = model.predict(future_df[['year']]).tolist()

    return {
        "historical": {
            "year": df['year'].tolist(),
            target_column: df[target_column].tolist()
        },
        "metrics": {
            "r2": score
        },
        "forecast": {
            "year": future_years,
            "pred": future_preds
        }
    }
