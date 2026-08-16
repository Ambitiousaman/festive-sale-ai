import pandas as pd

FEATURES = [
    "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28",
    "discount", "price_ratio", "competitor_ratio",
    "festive", "sale_event", "weekend", "dayofweek", "month"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["sku", "date"])
    g = df.groupby("sku", group_keys=False)

    df["lag_1"] = g["units_sold"].shift(1)
    df["lag_7"] = g["units_sold"].shift(7)
    df["lag_14"] = g["units_sold"].shift(14)
    df["rolling_7"] = g["units_sold"].transform(lambda s: s.shift(1).rolling(7).mean())
    df["rolling_28"] = g["units_sold"].transform(lambda s: s.shift(1).rolling(28).mean())

    df["price_ratio"] = df["price"] / df["list_price"]
    df["competitor_ratio"] = df["price"] / df["competitor_price"]

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.dropna().reset_index(drop=True)
