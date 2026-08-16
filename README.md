# Festive Sale AI — E-commerce Sale Intelligence Platform

A portfolio-grade simulation of the AI/ML decision system that a large e-commerce marketplace could use for a festive event such as Big Billion Days / Great Indian Festival.

> Important: Flipkart/Amazon do not publicly disclose their complete proprietary production algorithms. This project is an independent, reproducible architecture inspired by publicly described e-commerce techniques. Flipkart has publicly described forecasting + optimization layers, including statistical models, CatBoost, LSTM/Seq2Seq and stochastic optimization for supply-chain decisions. Alibaba has also described integrated demand forecasting, inventory optimization, price optimization and recommendations.

## What this project does

For every SKU and sale day it can:

1. Forecast demand
2. Estimate price elasticity
3. Recommend an economically sensible sale price
4. Calculate safety stock and replenishment quantity
5. Rank products for promotion
6. Generate personalized product recommendations
7. Expose predictions through FastAPI
8. Visualize results through Streamlit

## Algorithms

### Demand Forecasting
Baseline:
- lag features
- rolling averages
- event/holiday features
- price and discount features
- HistGradientBoostingRegressor

Production upgrade path:
- CatBoost / LightGBM
- LSTM / Seq2Seq
- probabilistic forecasting
- hierarchical SKU/category forecasting

### Price Optimization
Demand model:

    log(Q) = beta0 + beta1*log(P) + beta2*discount + beta3*event + ...

Price elasticity is approximately beta1.

The optimizer evaluates candidate prices and maximizes:

    expected_profit = (price - unit_cost) * expected_demand

subject to inventory and business constraints.

### Recommendations
Collaborative filtering:
- user-item interaction matrix
- TruncatedSVD matrix factorization
- cosine similarity
- popularity fallback for cold-start users

Production upgrade path:
- two-tower retrieval
- ANN/vector search
- learning-to-rank
- deep recommendation models

### Inventory
Safety stock:

    safety_stock = z * sigma_demand * sqrt(lead_time)

Reorder point:

    ROP = mean_demand * lead_time + safety_stock

The sale engine adds event demand uplift.

## Project architecture

                        ┌─────────────────────┐
                        │ Web / Mobile Events │
                        │ views/clicks/orders  │
                        └──────────┬──────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Event Stream      │
                         │ Kafka (production)│
                         └─────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ Data Lake / Warehouse       │
                    │ S3 + Spark + SQL            │
                    └──────────────┬──────────────┘
                                   │
                 ┌─────────────────▼─────────────────┐
                 │ Feature Engineering / Feature     │
                 │ Store                              │
                 └───────┬───────────┬──────────────┘
                         │           │
          ┌──────────────▼───┐   ┌──▼────────────────┐
          │ Demand Forecast  │   │ Recommendation     │
          │ ML model         │   │ Retrieval / Ranker │
          └──────────────┬───┘   └──────────┬─────────┘
                         │                  │
                 ┌───────▼──────────────────▼───────┐
                 │ Decision Engine                    │
                 │ Price + Inventory + Promotion     │
                 └──────────────┬────────────────────┘
                                │
                      ┌─────────▼─────────┐
                      │ FastAPI Serving   │
                      └─────────┬─────────┘
                                │
                       ┌────────▼────────┐
                       │ Streamlit UI    │
                       └─────────────────┘

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

python src/generate_data.py
python src/train.py

uvicorn app.api:app --reload
```

Open:
- API: http://127.0.0.1:8000/docs
- Dashboard: in another terminal, `streamlit run app/dashboard.py`

## Project structure

    festive_sale_ai/
    ├── app/
    │   ├── api.py
    │   └── dashboard.py
    ├── data/
    │   ├── products.csv
    │   ├── sales.csv
    │   └── interactions.csv
    ├── models/
    ├── src/
    │   ├── generate_data.py
    │   ├── features.py
    │   ├── demand_model.py
    │   ├── price_optimizer.py
    │   ├── inventory.py
    │   ├── recommender.py
    │   └── train.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

## Interview explanation

"The project is not simply a discount predictor. It is a decision system. Demand forecasting predicts how many units customers are likely to buy. The price-elasticity model estimates how demand changes with price. The optimizer tests feasible prices and selects the one that maximizes expected contribution while respecting inventory constraints. A separate recommendation model personalizes product exposure. Finally, the inventory module calculates safety stock and reorder points so the marketplace can position inventory before the sale."

## Important business insight

A ₹20,000 iPhone being sold for ₹15,000 does not necessarily mean the marketplace loses ₹5,000.

The final customer value can be a combination of:
- seller-funded discount
- bank-funded offer
- platform subsidy
- exchange value
- cashback
- advertising revenue
- accessory attach rate
- customer acquisition / lifetime value
- inventory and logistics economics

Therefore, the optimizer should model **net contribution**, not only displayed discount.

## Production improvements

- Replace synthetic data with historical marketplace data
- Use Spark for feature pipelines
- Use Kafka for real-time events
- Use MLflow for experiment tracking
- Use a feature store
- Use Redis for low-latency model features
- Use a vector database / ANN index for recommendation retrieval
- Add model monitoring and drift detection
- Add A/B testing
- Add probabilistic forecasting and stochastic optimization

## FestiKart AI E-commerce Demo

The project now includes a Flipkart-inspired (but independently branded) e-commerce demo UI.

Run:

```bash
python src/generate_data.py
python src/train.py
uvicorn app.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

The UI supports:
- Normal pricing vs Festive Sale mode
- Product catalogue
- AI demand forecast
- AI dynamic price optimization
- Inventory policy
- Before/after sale comparison
- Personalized recommendations
- Search and category filtering

The UI is served directly by FastAPI, so no separate frontend server is required.
