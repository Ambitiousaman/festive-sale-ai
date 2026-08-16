import sys
from pathlib import Path
from html import escape

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PATH SETUP
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_features, FEATURES
from src.price_optimizer import optimize_price
from src.inventory import inventory_policy


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Festive Sale AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ================================
       HEADER
       ================================ */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #202124;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* ================================
       FESTIVE SALE BANNER
       ================================ */

    .sale-banner {
        background: linear-gradient(
            135deg,
            #ff512f 0%,
            #f02d74 100%
        );

        padding: 30px;
        border-radius: 18px;

        color: white;
        text-align: center;

        margin: 20px 0 30px 0;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.12);
    }

    .sale-banner h1 {
        font-size: 36px;
        margin: 0;
        font-weight: 900;
    }

    .sale-banner p {
        font-size: 17px;
        margin: 8px 0 0 0;
    }

    /* ================================
       PRODUCT CARD
       ================================ */

    .product-card {
        background: white;

        padding: 25px;

        border-radius: 18px;

        border: 1px solid #e5e7eb;

        box-shadow:
            0 5px 20px rgba(0, 0, 0, 0.06);

        margin-bottom: 20px;
    }

    .product-name {
        font-size: 27px;
        font-weight: 800;
        color: #202124;
    }

    .product-meta {
        color: #6b7280;
        font-size: 15px;
        margin-top: 5px;
    }

    .price-row {
        margin-top: 18px;
    }

    .old-price {
        text-decoration: line-through;
        color: #888;
        font-size: 20px;
    }

    .sale-price {
        color: #d32f2f;
        font-size: 36px;
        font-weight: 900;
        margin-top: 8px;
    }

    .discount-badge {
        background: #008c45;
        color: white;

        padding: 6px 12px;

        border-radius: 7px;

        font-weight: 700;

        display: inline-block;
        margin-left: 10px;
    }

    /* ================================
       SECTION TITLE
       ================================ */

    .section-title {
        font-size: 27px;
        font-weight: 800;

        margin-top: 30px;
        margin-bottom: 15px;

        color: #202124;
    }

    /* ================================
       COMPARISON CARD
       ================================ */

    .comparison-card {
        background: white;

        padding: 22px;

        border-radius: 16px;

        border: 1px solid #e4e4e4;

        min-height: 125px;

        box-shadow:
            0 4px 15px rgba(0, 0, 0, 0.04);
    }

    .comparison-title {
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    /* ================================
       AI CARD
       ================================ */

    .ai-card {
        background: #eef5ff;

        border-left: 5px solid #2874f0;

        padding: 20px;

        border-radius: 12px;

        margin: 15px 0;
    }

    .ai-title {
        font-size: 20px;
        font-weight: 800;
        color: #174ea6;
    }

    .ai-text {
        font-size: 16px;
        line-height: 1.7;
        color: #374151;
    }

    /* ================================
       RECOMMENDATION CARD
       ================================ */

    .recommendation-card {
        background: white;

        border: 1px solid #e5e7eb;

        border-radius: 14px;

        padding: 15px 18px;

        margin: 8px 0;

        font-size: 16px;

        font-weight: 600;
    }

    /* ================================
       FOOTER
       ================================ */

    .footer {
        text-align: center;

        color: #888;

        padding: 35px 10px;

        font-size: 13px;
    }

    </style>
    """
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    sales = pd.read_csv(
        ROOT / "data" / "sales.csv",
        parse_dates=["date"]
    )

    products = pd.read_csv(
        ROOT / "data" / "products.csv"
    )

    return sales, products


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    demand_model = joblib.load(
        ROOT / "models" / "demand_model.joblib"
    )

    elasticities = joblib.load(
        ROOT / "models" / "elasticities.joblib"
    )

    recommender = joblib.load(
        ROOT / "models" / "recommender.joblib"
    )

    return demand_model, elasticities, recommender


# =========================================================
# LOAD EVERYTHING
# =========================================================

sales, products = load_data()

demand_model, elasticities, recommender = load_models()


# =========================================================
# HEADER
# =========================================================

st.html(
    """
    <div class="main-title">
        🛒 Festive Sale AI Decision Platform
    </div>

    <div class="subtitle">
        AI-powered e-commerce platform for demand forecasting,
        dynamic pricing, inventory planning and personalized recommendations.
    </div>
    """
)


# =========================================================
# FESTIVE BANNER
# =========================================================

st.html(
    """
    <div class="sale-banner">

        <h1>🔥 BIG FESTIVE DAYS 🔥</h1>

        <p>
            AI-Powered Deals • Smart Pricing • Limited-Time Offers
        </p>

    </div>
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Configuration")

sku = st.sidebar.selectbox(
    "Select Product",
    products["sku"].tolist()
)

product = products[
    products["sku"] == sku
].iloc[0]

product_sales = sales[
    sales["sku"] == sku
].copy()

st.sidebar.markdown("---")

st.sidebar.subheader("💰 Pricing Inputs")

inventory = st.sidebar.number_input(
    "Available Inventory",
    min_value=0.0,
    value=float(product["initial_inventory"]),
    step=1.0
)

competitor_price = st.sidebar.number_input(
    "Competitor Price",
    min_value=1.0,
    value=float(product["list_price"] * 0.90),
    step=100.0
)


# =========================================================
# DEMAND FORECAST
# =========================================================

features = build_features(product_sales)

# VERY IMPORTANT:
# Make prediction dataframe exactly match training features.

features = features.reindex(
    columns=FEATURES,
    fill_value=0
)

forecast = float(
    demand_model.predict(
        features.tail(1)
    )[0]
)

elasticity = float(
    elasticities.get(
        sku,
        -1.2
    )
)


# =========================================================
# PRICE OPTIMIZATION
# =========================================================

optimization = optimize_price(
    float(product["list_price"]),
    float(product["unit_cost"]),
    forecast,
    elasticity,
    inventory,
    competitor_price=competitor_price
)

recommended_price = float(
    optimization["recommended_price"]
)

discount_pct = float(
    optimization["recommended_discount_pct"]
)

expected_units = float(
    optimization["expected_units"]
)

expected_profit = float(
    optimization["expected_profit"]
)


# =========================================================
# BASIC VALUES
# =========================================================

list_price = float(
    product["list_price"]
)

unit_cost = float(
    product["unit_cost"]
)

before_price = list_price

before_units = forecast

before_profit = max(
    0,
    (before_price - unit_cost) * before_units
)


# =========================================================
# SALE DEMAND IMPACT
# =========================================================

if before_price > 0:

    estimated_sale_units = (
        forecast
        * (recommended_price / before_price)
        ** elasticity
    )

else:

    estimated_sale_units = forecast


estimated_sale_units = min(
    max(estimated_sale_units, 0),
    inventory
)


# =========================================================
# PRODUCT OVERVIEW
# =========================================================

st.html(
    """
    <div class="section-title">
        📦 Product Overview
    </div>
    """
)

brand = escape(str(product["brand"]))
safe_sku = escape(str(sku).replace("_", " "))
category = escape(str(product["category"]))

product_html = f"""
<div class="product-card">

    <div class="product-name">
        {brand} {safe_sku}
    </div>

    <div class="product-meta">
        Category: {category}
    </div>

    <div class="price-row">

        <span class="old-price">
            ₹{list_price:,.0f}
        </span>

        <span class="discount-badge">
            AI SALE {discount_pct:.1f}% OFF
        </span>

        <div class="sale-price">
            ₹{recommended_price:,.0f}
        </div>

    </div>

</div>
"""

st.html(product_html)


# =========================================================
# KEY METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Next-Day Demand",
    f"{forecast:.0f} units"
)

c2.metric(
    "List Price",
    f"₹{list_price:,.0f}"
)

c3.metric(
    "Unit Cost",
    f"₹{unit_cost:,.0f}"
)

c4.metric(
    "Price Elasticity",
    f"{elasticity:.2f}"
)


# =========================================================
# SALE SIMULATOR
# =========================================================

st.html(
    """
    <div class="section-title">
        🔥 Sale Simulator
    </div>
    """
)

sale_started = st.toggle(
    "Activate Festive Sale",
    value=False
)


# =========================================================
# BEFORE VS FESTIVE SALE
# =========================================================

st.html(
    """
    <div class="section-title">
        📊 Before Sale vs Festive Sale
    </div>
    """
)

if sale_started:

    st.success(
        "🔥 Festive Sale is LIVE — AI pricing has been activated!"
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # BEFORE SALE
    # -----------------------------------------------------

    with col1:

        st.html(
            """
            <div class="comparison-card">

                <div class="comparison-title">
                    🛍️ Before Sale
                </div>

            </div>
            """
        )

        st.metric(
            "Selling Price",
            f"₹{before_price:,.0f}"
        )

        st.metric(
            "Discount",
            "0%"
        )

        st.metric(
            "Expected Demand",
            f"{before_units:.0f} units"
        )

        st.metric(
            "Expected Profit",
            f"₹{before_profit:,.0f}"
        )

    # -----------------------------------------------------
    # FESTIVE SALE
    # -----------------------------------------------------

    with col2:

        st.html(
            """
            <div class="comparison-card">

                <div class="comparison-title">
                    🔥 Festive Sale
                </div>

            </div>
            """
        )

        st.metric(
            "AI Sale Price",
            f"₹{recommended_price:,.0f}",
            delta=f"-₹{before_price - recommended_price:,.0f}"
        )

        st.metric(
            "Discount",
            f"{discount_pct:.1f}%"
        )

        st.metric(
            "Expected Demand",
            f"{estimated_sale_units:.0f} units",
            delta=f"{estimated_sale_units - before_units:+.0f}"
        )

        st.metric(
            "Expected Profit",
            f"₹{expected_profit:,.0f}",
            delta=f"₹{expected_profit - before_profit:,.0f}"
        )

else:

    st.info(
        "Turn on **Activate Festive Sale** to simulate "
        "the AI-powered festive pricing impact."
    )


# =========================================================
# AI PRICE OPTIMIZATION
# =========================================================

st.html(
    """
    <div class="section-title">
        🤖 AI Price Optimization
    </div>
    """
)

a, b, c, d = st.columns(4)

a.metric(
    "Recommended Sale Price",
    f"₹{recommended_price:,.0f}"
)

b.metric(
    "Discount",
    f"{discount_pct:.1f}%"
)

c.metric(
    "Expected Units",
    f"{expected_units:.0f}"
)

d.metric(
    "Expected Profit",
    f"₹{expected_profit:,.0f}"
)


# =========================================================
# AI PRICING EXPLANATION
# =========================================================

safe_competitor = f"₹{competitor_price:,.0f}"
safe_recommended = f"₹{recommended_price:,.0f}"

st.html(
    f"""
    <div class="ai-card">

        <div class="ai-title">
            🧠 AI Pricing Decision
        </div>

        <br>

        <div class="ai-text">

            Based on historical demand, price elasticity,
            inventory and competitor pricing, the AI recommends:

            <br><br>

            <strong style="font-size:24px;">
                {safe_recommended}
            </strong>

            as the optimal festive-sale price.

            <br><br>

            Current competitor price:
            <strong>{safe_competitor}</strong>

            <br>

            Estimated elasticity:
            <strong>{elasticity:.2f}</strong>

        </div>

    </div>
    """
)


# =========================================================
# SALE IMPACT
# =========================================================

st.html(
    """
    <div class="section-title">
        📈 Sale Impact
    </div>
    """
)

demand_change = (
    estimated_sale_units - before_units
)

price_change = (
    recommended_price - before_price
)

profit_change = (
    expected_profit - before_profit
)

impact1, impact2, impact3 = st.columns(3)

impact1.metric(
    "Demand Change",
    f"{demand_change:+.0f} units"
)

impact2.metric(
    "Price Change",
    f"₹{price_change:,.0f}"
)

impact3.metric(
    "Profit Change",
    f"₹{profit_change:,.0f}"
)


# =========================================================
# INVENTORY PLANNING
# =========================================================

st.html(
    """
    <div class="section-title">
        📦 Inventory Planning
    </div>
    """
)

mean_demand = product_sales["units_sold"].mean()

std_demand = product_sales["units_sold"].std()

policy = inventory_policy(
    mean_demand,
    std_demand,
    5
)


if isinstance(policy, dict):

    safety_stock = policy.get(
        "safety_stock",
        policy.get(
            "safety_stock_units",
            0
        )
    )

    reorder_point = policy.get(
        "reorder_point",
        0
    )

    reorder_qty = policy.get(
        "suggested_reorder_qty",
        policy.get(
            "reorder_quantity",
            0
        )
    )

else:

    safety_stock = 0
    reorder_point = 0
    reorder_qty = 0


inv1, inv2, inv3 = st.columns(3)

inv1.metric(
    "Safety Stock",
    f"{float(safety_stock):.0f} units"
)

inv2.metric(
    "Reorder Point",
    f"{float(reorder_point):.0f} units"
)

inv3.metric(
    "Suggested Reorder",
    f"{float(reorder_qty):.0f} units"
)


# =========================================================
# INVENTORY STATUS
# =========================================================

if inventory < reorder_point:

    st.warning(
        f"⚠️ Inventory ({inventory:.0f}) is below the "
        f"recommended reorder point ({float(reorder_point):.0f})."
    )

else:

    st.success(
        f"✅ Inventory level is currently healthy "
        f"({inventory:.0f} units available)."
    )


# =========================================================
# DEMAND HISTORY
# =========================================================

st.html(
    """
    <div class="section-title">
        📈 Demand History
    </div>
    """
)

chart = (
    product_sales
    .sort_values("date")
    .set_index("date")["units_sold"]
    .tail(120)
)

st.line_chart(chart)


# =========================================================
# PERSONALIZED RECOMMENDATIONS
# =========================================================

st.html(
    """
    <div class="section-title">
        🎯 Personalized Recommendations
    </div>
    """
)

user = st.text_input(
    "Enter User ID",
    value="U0001"
)

if st.button(
    "Generate Recommendations",
    type="primary"
):

    try:

        recommendations = recommender.recommend(
            user,
            5
        )

        st.success(
            f"Recommendations generated for {user}"
        )

        # Handle list/array responses nicely

        if isinstance(
            recommendations,
            (list, tuple)
        ):

            for i, item in enumerate(
                recommendations,
                start=1
            ):

                st.html(
                    f"""
                    <div class="recommendation-card">
                        #{i} &nbsp; 🛍️ {escape(str(item))}
                    </div>
                    """
                )

        else:

            st.write(
                recommendations
            )

    except Exception as e:

        st.error(
            f"Could not generate recommendations: {e}"
        )


# =========================================================
# AI DECISION SUMMARY
# =========================================================

st.html(
    """
    <div class="section-title">
        🧠 AI Decision Summary
    </div>
    """
)

decision = (
    "🔥 Activate festive pricing"
    if sale_started
    else
    "⏳ Waiting for sale activation"
)

safe_decision = escape(decision)

st.html(
    f"""
    <div class="ai-card">

        <div class="ai-title">
            AI Business Decision
        </div>

        <br>

        <div class="ai-text">

            <strong>Product:</strong>
            {escape(str(sku))}

            <br><br>

            <strong>Forecasted next-day demand:</strong>
            {forecast:.0f} units

            <br>

            <strong>Current list price:</strong>
            ₹{list_price:,.0f}

            <br>

            <strong>AI recommended festive price:</strong>
            ₹{recommended_price:,.0f}

            <br>

            <strong>Recommended discount:</strong>
            {discount_pct:.1f}%

            <br>

            <strong>Expected demand during sale:</strong>
            {estimated_sale_units:.0f} units

            <br>

            <strong>Expected contribution:</strong>
            ₹{expected_profit:,.0f}

            <br><br>

            <strong>Decision:</strong>
            {safe_decision}

        </div>

    </div>
    """
)


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">

        Festive Sale AI Decision Platform

        <br>

        Synthetic E-Commerce Analytics Project

        <br><br>

        Demand Forecasting • Dynamic Pricing •
        Price Elasticity • Inventory Optimization •
        Personalized Recommendations

    </div>
    """
)