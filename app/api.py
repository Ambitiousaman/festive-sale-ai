from pathlib import Path
import sys
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import build_features, FEATURES
from src.price_optimizer import optimize_price
from src.inventory import inventory_policy

app = FastAPI(title="Festive Sale AI API", version="1.0")

DATA = ROOT / "data"
MODELS = ROOT / "models"

# These files are created by:
#   python src/generate_data.py
#   python src/train.py
try:
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["date"])
    products = pd.read_csv(DATA / "products.csv")
    demand_model = joblib.load(MODELS / "demand_model.joblib")
    elasticities = joblib.load(MODELS / "elasticities.joblib")
    recommender = joblib.load(MODELS / "recommender.joblib")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Required data/model file is missing: {exc}. "
        "Run 'python src/generate_data.py' and then 'python src/train.py'."
    ) from exc


class ForecastRequest(BaseModel):
    sku: str


class PriceRequest(BaseModel):
    sku: str
    inventory: float = Field(default=500, ge=0)
    competitor_price: float | None = Field(default=None, gt=0)


class RecommendRequest(BaseModel):
    user_id: str
    n: int = Field(default=5, ge=1, le=20)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api")
def api_root():
    return {"service": "Festive Sale AI", "status": "running"}


@app.get("/products")
def get_products():
    cols = [
        "sku", "brand", "category", "list_price",
        "unit_cost", "initial_inventory"
    ]
    available = [c for c in cols if c in products.columns]
    return products[available].fillna(0).to_dict(orient="records")


def _latest_features(sku: str):
    s = sales[sales["sku"] == sku].copy()

    if s.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")

    f = build_features(s)

    if f.empty:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough historical data to forecast SKU '{sku}'"
        )

    # IMPORTANT:
    # The trained model expects exactly these 13 columns in this order.
    X = f.tail(1)[FEATURES].copy()
    return s, f, X


@app.post("/forecast")
def forecast(req: ForecastRequest):
    _, _, X = _latest_features(req.sku)

    pred = float(demand_model.predict(X)[0])
    pred = max(0.0, pred)

    return {
        "sku": req.sku,
        "forecast_units_next_day": round(pred, 2),
    }


@app.post("/optimize-price")
def optimize(req: PriceRequest):
    product_rows = products[products["sku"] == req.sku]

    if product_rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{req.sku}' not found")

    p = product_rows.iloc[0]
    s, _, X = _latest_features(req.sku)

    predicted_demand = max(0.0, float(demand_model.predict(X)[0]))
    elasticity = float(elasticities.get(req.sku, -1.2))

    competitor_price = req.competitor_price
    if competitor_price is None:
        latest_comp = s["competitor_price"].dropna()
        competitor_price = (
            float(latest_comp.iloc[-1])
            if not latest_comp.empty
            else float(p["list_price"]) * 0.90
        )

    result = optimize_price(
        float(p["list_price"]),
        float(p["unit_cost"]),
        predicted_demand,
        elasticity,
        req.inventory,
        competitor_price=competitor_price,
    )

    result["sku"] = req.sku
    result["list_price"] = float(p["list_price"])
    result["unit_cost"] = float(p["unit_cost"])
    result["forecast_units_next_day"] = round(predicted_demand, 2)
    result["competitor_price"] = round(float(competitor_price), 2)

    return result


@app.post("/inventory")
def inventory(req: ForecastRequest):
    s = sales[sales["sku"] == req.sku]["units_sold"]

    if s.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{req.sku}' not found")

    policy = inventory_policy(
        float(s.mean()),
        float(s.std()),
        lead_time_days=5
    )
    policy["sku"] = req.sku
    return policy


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return {
        "user_id": req.user_id,
        "recommendations": recommender.recommend(req.user_id, req.n),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "products": len(products),
        "sales_rows": len(sales),
    }
