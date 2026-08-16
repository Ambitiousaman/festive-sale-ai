let products = [];
let saleMode = false;
let selectedSku = null;
let cartCount = 0;

const money = value => "₹" + Number(value || 0).toLocaleString("en-IN", {
  maximumFractionDigits: 0
});

const api = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: {"Content-Type": "application/json"},
    ...options
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return body;
};

function toast(message) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2500);
}

function productEmoji(category) {
  if (category === "Smartphones") return "📱";
  if (category === "Laptops") return "💻";
  if (category === "Audio") return "🎧";
  return "🛍️";
}

function productName(p) {
  return p.sku.replaceAll("_", " ");
}

async function loadProducts() {
  products = await api("/products");
  selectedSku = products[0]?.sku || null;

  const select = document.getElementById("skuSelect");
  select.innerHTML = products.map(p =>
    `<option value="${p.sku}">${productName(p)}</option>`
  ).join("");
  select.value = selectedSku;

  const p = products.find(x => x.sku === selectedSku);
  if (p) {
    document.getElementById("inventoryInput").value = p.initial_inventory || 500;
    document.getElementById("competitorInput").value = Math.round(p.list_price * .90);
  }

  renderProducts(products);
  await runAI(false);
  await loadRecommendations();
}

function renderProducts(list) {
  const grid = document.getElementById("productsGrid");
  grid.innerHTML = list.map(p => {
    const discount = saleMode && p.ai ? p.ai.recommended_discount_pct : 0;
    const currentPrice = saleMode && p.ai ? p.ai.recommended_price : p.list_price;
    return `
      <article class="product" onclick="selectProduct('${p.sku}')">
        <div class="product-img">${productEmoji(p.category)}</div>
        <div class="brandline">${p.brand} · ${p.category}</div>
        <h3>${productName(p)}</h3>
        ${saleMode && p.ai ? `<div class="old">${money(p.list_price)}</div>` : ""}
        <div class="price">${money(currentPrice)}
          ${saleMode && p.ai ? `<span class="discount">${discount.toFixed(1)}% OFF</span>` : ""}
        </div>
        ${saleMode && p.ai
          ? `<div class="sale-badge">🔥 AI FESTIVE PRICE</div>`
          : `<div class="mini">Tap to run AI pricing</div>`}
      </article>
    `;
  }).join("");
}

async function selectProduct(sku) {
  selectedSku = sku;
  document.getElementById("skuSelect").value = sku;
  const p = products.find(x => x.sku === sku);
  if (p) {
    document.getElementById("inventoryInput").value = p.initial_inventory || 500;
    document.getElementById("competitorInput").value = Math.round(p.list_price * .90);
  }
  await runAI(false);
  document.querySelector(".dashboard").scrollIntoView({behavior:"smooth"});
}

async function runAI(showToast = true) {
  selectedSku = document.getElementById("skuSelect").value;
  const p = products.find(x => x.sku === selectedSku);
  if (!p) return;

  try {
    const inventory = Number(document.getElementById("inventoryInput").value || 0);
    const competitor = Number(document.getElementById("competitorInput").value || p.list_price * .9);

    const [forecast, price, inv] = await Promise.all([
      api("/forecast", {method:"POST", body:JSON.stringify({sku:selectedSku})}),
      api("/optimize-price", {
        method:"POST",
        body:JSON.stringify({
          sku:selectedSku,
          inventory,
          competitor_price:competitor
        })
      }),
      api("/inventory", {method:"POST", body:JSON.stringify({sku:selectedSku})})
    ]);

    p.ai = price;

    document.getElementById("mDemand").textContent = forecast.forecast_units_next_day.toFixed(1);
    document.getElementById("mPrice").textContent = money(price.recommended_price);
    document.getElementById("mDiscount").textContent = price.recommended_discount_pct.toFixed(1) + "%";
    document.getElementById("mUnits").textContent = price.expected_units.toFixed(1);
    document.getElementById("mProfit").textContent = money(price.expected_profit);
    document.getElementById("mElasticity").textContent = price.elasticity.toFixed(2);

    document.getElementById("impactBefore").textContent = money(p.list_price);
    document.getElementById("impactAfter").textContent = saleMode
      ? money(price.recommended_price)
      : money(p.list_price);
    document.getElementById("impactDiscount").textContent = saleMode
      ? price.recommended_discount_pct.toFixed(1) + "% AI discount"
      : "Activate sale";
    document.getElementById("impactDemand").textContent =
      forecast.forecast_units_next_day.toFixed(1);
    document.getElementById("impactProfit").textContent =
      money(price.expected_profit);

    document.getElementById("heroDemand").textContent =
      forecast.forecast_units_next_day.toFixed(0);

    document.getElementById("whyPrice").textContent =
      `The model forecasts ${forecast.forecast_units_next_day.toFixed(1)} units/day. ` +
      `Using elasticity ${price.elasticity.toFixed(2)}, competitor price ${money(price.competitor_price)}, ` +
      `inventory ${inventory.toLocaleString("en-IN")} units and the product cost, ` +
      `the optimizer selects ${money(price.recommended_price)} to maximize expected contribution ` +
      `within the configured discount range. Safety-stock reorder point is about ${inv.reorder_point.toFixed(0)} units.`;

    renderProducts(products);
    if (showToast) toast("AI decision engine completed.");
  } catch (error) {
    toast(error.message);
    console.error(error);
  }
}

async function toggleSale() {
  saleMode = !saleMode;
  const hero = document.getElementById("saleHero");
  const button = document.getElementById("saleToggle");
  const status = document.getElementById("saleStatus");

  if (saleMode) {
    hero.classList.remove("normal");
    hero.classList.add("sale");
    document.getElementById("heroTitle").textContent = "🔥 FESTIVE SALE IS LIVE!";
    document.getElementById("heroText").textContent =
      "AI is dynamically optimizing prices using demand, elasticity, competitor pricing and inventory.";
    button.textContent = "✓ Festive Sale Active — Turn Off";
    status.textContent = "🔥 AI SALE LIVE";
    status.classList.add("live");

    await runAI(false);
    toast("Festive Sale activated — AI prices are now live.");
  } else {
    hero.classList.remove("sale");
    hero.classList.add("normal");
    document.getElementById("heroTitle").textContent = "Smart shopping starts here.";
    document.getElementById("heroText").textContent =
      "AI-powered demand forecasting, pricing and inventory decisions.";
    button.textContent = "🔥 Activate Festive Sale";
    status.textContent = "NORMAL PRICING";
    status.classList.remove("live");
    renderProducts(products);
    await runAI(false);
    toast("Sale mode turned off.");
  }
}

async function loadRecommendations() {
  const userId = document.getElementById("userId").value || "U0001";
  try {
    const data = await api("/recommend", {
      method:"POST",
      body:JSON.stringify({user_id:userId, n:5})
    });
    document.getElementById("recommendationsGrid").innerHTML =
      data.recommendations.map(sku =>
        `<div class="rec">🛍️ ${productName({sku})}</div>`
      ).join("");
  } catch (error) {
    toast(error.message);
  }
}

document.getElementById("skuSelect").addEventListener("change", runAI);
document.getElementById("searchInput").addEventListener("input", e => {
  const q = e.target.value.toLowerCase().trim();
  const filtered = products.filter(p =>
    productName(p).toLowerCase().includes(q) ||
    p.brand.toLowerCase().includes(q) ||
    p.category.toLowerCase().includes(q)
  );
  renderProducts(filtered);
});

document.querySelectorAll(".categories button").forEach(btn => {
  btn.addEventListener("click", () => {
    const category = btn.dataset.category;
    renderProducts(category === "All"
      ? products
      : products.filter(p => p.category === category));
  });
});

function scrollToTop() {
  window.scrollTo({top:0, behavior:"smooth"});
}

document.addEventListener("DOMContentLoaded", () => {
  loadProducts().catch(error => {
    toast(error.message);
    console.error(error);
  });
});
