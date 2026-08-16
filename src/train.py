import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import build_features
from src.demand_model import DemandModel
from src.price_optimizer import estimate_elasticity
from src.recommender import Recommender

DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

sales = pd.read_csv(DATA / "sales.csv", parse_dates=["date"])
featured = build_features(sales)

# Time-based split prevents future leakage.
cutoff = featured["date"].quantile(0.82)
train = featured[featured["date"] <= cutoff]
test = featured[featured["date"] > cutoff]

model = DemandModel().fit(train)
metrics = model.evaluate(test)
model.save(MODELS / "demand_model.joblib")

# Category/SKU-level elasticity estimates.
elasticities = {}
for sku, g in sales.groupby("sku"):
    elasticities[sku] = estimate_elasticity(g)
joblib.dump(elasticities, MODELS / "elasticities.joblib")

interactions = pd.read_csv(DATA / "interactions.csv")
rec = Recommender().fit(interactions)
rec.save(MODELS / "recommender.joblib")

print("Demand model metrics:", metrics)
print("Saved models to:", MODELS)
print("Example elasticity:", list(elasticities.items())[:3])
