import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

products = [
    ("IPHONE_15_128", "Apple", "Smartphones", 70000, 52000),
    ("IPHONE_15_256", "Apple", "Smartphones", 80000, 60000),
    ("IPHONE_16_128", "Apple", "Smartphones", 80000, 61000),
    ("IPHONE_16_256", "Apple", "Smartphones", 90000, 69000),
    ("SAMSUNG_S24", "Samsung", "Smartphones", 75000, 57000),
    ("ONEPLUS_13", "OnePlus", "Smartphones", 70000, 54000),
    ("MACBOOK_AIR_M2", "Apple", "Laptops", 100000, 82000),
    ("DELL_XPS_13", "Dell", "Laptops", 115000, 92000),
    ("SONY_HEADPHONES", "Sony", "Audio", 30000, 21000),
    ("AIRPODS_PRO", "Apple", "Audio", 25000, 17000),
]

product_df = pd.DataFrame(
    products, columns=["sku", "brand", "category", "list_price", "unit_cost"]
)
product_df["base_demand"] = [55, 35, 70, 45, 50, 60, 25, 18, 80, 95]
product_df["initial_inventory"] = [900, 600, 1100, 700, 850, 950, 500, 350, 1200, 1400]
product_df.to_csv(DATA / "products.csv", index=False)

dates = pd.date_range("2025-01-01", "2026-07-31", freq="D")
rows = []

for _, p in product_df.iterrows():
    for d in dates:
        festive = int((d.month in [9, 10, 11]) and d.day in list(range(1, 8)) + list(range(20, 31)))
        sale_event = int(d in pd.date_range("2025-10-06", "2025-10-12") or
                         d in pd.date_range("2026-10-01", "2026-10-07"))
        weekend = int(d.dayofweek >= 5)
        trend = 1 + 0.00035 * (d - dates[0]).days
        season = 1 + 0.12 * np.sin(2 * np.pi * d.dayofyear / 365.25)
        event_uplift = 1 + 1.6 * sale_event + 0.35 * festive + 0.08 * weekend
        discount = np.clip(np.random.normal(0.08 + 0.10 * sale_event, 0.035), 0, 0.45)
        price = p["list_price"] * (1 - discount)
        competitor_price = price * np.random.normal(1.01, 0.025)

        # Synthetic demand-generating process.
        # Real marketplace data should replace this.
        elasticity = -1.35 if p["category"] == "Smartphones" else -1.05
        price_effect = (price / p["list_price"]) ** elasticity
        lam = p["base_demand"] * trend * season * event_uplift * price_effect
        units = np.random.poisson(max(lam, 0.1))

        rows.append([
            d, p["sku"], p["brand"], p["category"], p["list_price"],
            p["unit_cost"], price, competitor_price, discount,
            festive, sale_event, weekend, units
        ])

sales = pd.DataFrame(rows, columns=[
    "date","sku","brand","category","list_price","unit_cost","price",
    "competitor_price","discount","festive","sale_event","weekend","units_sold"
])
sales.to_csv(DATA / "sales.csv", index=False)

# User-item interactions for recommender.
users = [f"U{i:04d}" for i in range(1, 801)]
interaction_rows = []
for u in users:
    chosen = np.random.choice(product_df.sku, size=np.random.randint(2, 6), replace=False)
    for sku in chosen:
        interaction_rows.append([u, sku, np.random.choice([1, 2, 3, 4, 5], p=[.15,.2,.25,.25,.15])])

interactions = pd.DataFrame(interaction_rows, columns=["user_id","sku","rating"])
interactions.to_csv(DATA / "interactions.csv", index=False)

print("Generated:")
print(DATA / "products.csv")
print(DATA / "sales.csv")
print(DATA / "interactions.csv")
