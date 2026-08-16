import numpy as np

def estimate_elasticity(df):
    """
    Simple log-log elasticity estimate.
    beta_price < 0 means demand decreases as price rises.
    """
    x = np.log(df["price"].clip(lower=1))
    y = np.log(df["units_sold"].clip(lower=1))
    beta = np.polyfit(x, y, 1)[0]
    return float(beta)

def optimize_price(
    list_price,
    unit_cost,
    predicted_demand,
    elasticity=-1.2,
    inventory=500,
    min_discount=0.05,
    max_discount=0.35,
    competitor_price=None
):
    prices = np.linspace(
        list_price * (1 - max_discount),
        list_price * (1 - min_discount),
        80
    )

    # Price elasticity approximation around current prediction.
    base_price = list_price * 0.90
    demand = predicted_demand * (prices / base_price) ** elasticity

    # Competitive pressure.
    if competitor_price is not None:
        demand *= np.exp(-1.5 * ((prices / competitor_price) - 1))

    demand = np.minimum(demand, inventory)
    profit = (prices - unit_cost) * demand

    i = int(np.argmax(profit))
    return {
        "recommended_price": float(prices[i]),
        "recommended_discount_pct": float((1 - prices[i] / list_price) * 100),
        "expected_units": float(demand[i]),
        "expected_profit": float(profit[i]),
        "elasticity": float(elasticity)
    }
