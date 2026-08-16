import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.users = None
        self.items = None
        self.matrix = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions):
        pivot = interactions.pivot_table(
            index="user_id", columns="sku", values="rating", fill_value=0
        )
        self.users = pivot.index.tolist()
        self.items = pivot.columns.tolist()
        self.matrix = pivot.values

        k = min(self.n_components, max(2, min(self.matrix.shape) - 1))
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.user_factors = svd.fit_transform(self.matrix)
        self.item_factors = svd.components_.T
        self.svd = svd
        return self

    def recommend(self, user_id, n=5):
        if user_id not in self.users:
            # Cold-start fallback: popularity.
            popularity = np.asarray(self.matrix.sum(axis=0)).ravel()
            idx = np.argsort(-popularity)[:n]
            return [self.items[i] for i in idx]

        uidx = self.users.index(user_id)
        scores = self.user_factors[uidx] @ self.item_factors.T

        seen = self.matrix[uidx] > 0
        scores[seen] = -np.inf
        idx = np.argsort(-scores)[:n]
        return [self.items[i] for i in idx]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
