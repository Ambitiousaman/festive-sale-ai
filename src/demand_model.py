import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import FEATURES

class DemandModel:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.055, max_leaf_nodes=31,
            l2_regularization=0.2, random_state=42
        )

    def fit(self, train_df):
        X = train_df[FEATURES]
        y = train_df["units_sold"]
        self.model.fit(X, y)
        return self

    def predict(self, df):
        return self.model.predict(df[FEATURES]).clip(min=0)

    def evaluate(self, test_df):
        pred = self.predict(test_df)
        mae = mean_absolute_error(test_df["units_sold"], pred)
        rmse = mean_squared_error(test_df["units_sold"], pred) ** 0.5
        return {"MAE": float(mae), "RMSE": float(rmse)}

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
