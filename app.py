import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FairDrive", layout="wide")
NOW, ERR = 2026, 200_000
st.markdown("""<style>
.card{background:linear-gradient(135deg,rgba(79,140,255,.18),rgba(255,255,255,.04));
border:1px solid rgba(255,255,255,.14);border-radius:16px;padding:18px 20px}
.card h4{margin:0;font-size:.75rem;color:#9fb0d0;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
.card p{margin:6px 0 0;font-size:1.65rem;font-weight:700;color:#fff}
.card small{color:#9fb0d0}
.block-container{padding-top:2rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_resource
def load_market():
    d = pd.read_csv("CAR_DETAILS_FROM_CAR_DEKHO.csv").drop_duplicates().dropna()
    d["car_age"] = NOW - d["year"]
    d["brand"] = d["name"].str.split().str[0]
    d = d[d["selling_price"] < d["selling_price"].quantile(0.99)]
    d = d[d["km_driven"] < 500000].reset_index(drop=True)
    cols = ["selling_price", "car_age", "km_driven"]
    sc = StandardScaler().fit(d[cols])
    km = KMeans(3, random_state=42, n_init=10).fit(sc.transform(d[cols]))
    order = d.groupby(km.labels_)["selling_price"].mean().sort_values().index
    names = dict(zip(order, ["Budget", "Mid-range", "Premium"]))
    d["segment"] = [names[i] for i in km.labels_]
    return d, sc, km, names


model, (mkt, scaler, kmeans, seg_names) = load_model(), load_market()
inr = lambda x: f"₹{x:,.0f}"
lakh = lambda x: f"₹{x / 1e5:.2f} L"
BRANDS = sorted(mkt["brand"].unique())
SHOW = {"Land": "Land Rover", "OpelCorsa": "Opel Corsa"}

# ---------------- Sidebar inputs (prices update live) ----------------
with st.sidebar:
    st.title("FairDrive")
    st.caption("Configure the car. Everything updates live.")
    brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti"),
                         format_func=lambda b: SHOW.get(b, b))
    year = st.number_input("Year", 1992, 2020, 2018, help="The data covers 1992 to 2020 models.")
    km = st.number_input("Kilometres driven", 0, 500000, 50000, step=5000)
    fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "LPG"])
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner",
                                   "Fourth & Above Owner", "Test Drive Car"])
    seller = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    asking = st.number_input("Seller's asking price (₹, 0 = skip)", 0, value=0, step=10000)

age = NOW - year
base = dict(km_driven=km, car_age=age, fuel=fuel, seller_type=seller,
            transmission=trans, owner=owner, brand=brand)


def predict(**over):
    rows = pd.DataFrame([{**base, **over}]) if not any(
        isinstance(v, (list, range, np.ndarray)) for v in over.values()) else None
    if rows is None:
        k, v = next(iter(over.items()))
        rows = pd.DataFrame([{**base, k: x} for x in v])
    return np.maximum(model.predict(rows), 20000)


price = float(predict()[0])
low, high = max(price - ERR, 0), price + ERR
seg = seg_names[int(kmeans.predict(scaler.transform([[price, age, km]]))[0])]

# ---------------- Header + KPI cards ----------------
st.markdown("## FairDrive · Used Car Fair-Price Dashboard")
if asking > 0:
    verdict = "High" if asking > high else "Low" if asking < low else "Fair"
    vsub = f"{(asking - price) / price:+.0%} vs estimate"
else:
    verdict, vsub = "—", "enter an asking price"
cards = [("Estimated fair price", lakh(price), inr(price)),
         ("Likely range", f"{lakh(low)} – {lakh(high)}", "typical error ≈ ₹2 L"),
         ("Market segment", seg, f"{age} years old · {km:,} km"),
         ("Asking price verdict", verdict, vsub)]
for col, (t, v, s) in zip(st.columns(4), cards):
    col.markdown(f"<div class='card'><h4>{t}</h4><p>{v}</p><small>{s}</small></div>",
                 unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs(["Estimate", "3D Market Map",
                                  "Similar Listings", "Model Performance"])

with tab1:
    g, c1 = st.columns([1, 1.4])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=price / 1e5, number={"suffix": " L", "valueformat": ".2f"},
        gauge={"axis": {"range": [0, 27]}, "bar": {"color": "#4f8cff"},
               "steps": [{"range": [low / 1e5, high / 1e5], "color": "rgba(255,255,255,.18)"}]}))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(t=30, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", title="Fair price (lakh ₹)")
    g.plotly_chart(fig, use_container_width=True)
    ages = list(range(6, 31))
    dep = pd.DataFrame({"Car age (years)": ages, "Price": predict(car_age=ages) / 1e5})
    f2 = px.area(dep, x="Car age (years)", y="Price", title="Depreciation: value vs age (lakh ₹)")
    f2.add_vline(x=age, line_dash="dot")
    f2.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)")
    c1.plotly_chart(f2, use_container_width=True)
    kms = list(range(0, 200001, 10000))
    kd = pd.DataFrame({"Kilometres": kms, "Price": predict(km_driven=kms) / 1e5})
    f3 = px.line(kd, x="Kilometres", y="Price", markers=True,
                 title="Value vs kilometres driven (lakh ₹)")
    f3.update_layout(template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f3, use_container_width=True)
    drop = float(predict(km_driven=[km])[0] - predict(km_driven=[km + 25000])[0])
    # st.info(f"Every extra 25,000 km lowers this car's value by about {inr(max(drop, 0))}.")

with tab2:
    st.caption("Every point is a real listing: age, kilometres and price. K-Means groups them into "
               "Budget, Mid-range and Premium. Drag to rotate, scroll to zoom. Your car is the gold diamond.")
    s = mkt.sample(min(1500, len(mkt)), random_state=1)
    f4 = px.scatter_3d(s, x="car_age", y="km_driven", z="selling_price", color="segment",
                       opacity=.55, labels={"car_age": "Age (yrs)", "km_driven": "Km",
                                            "selling_price": "Price (₹)"},
                       color_discrete_map={"Budget": "#3b82f6", "Mid-range": "#f59e0b",
                                           "Premium": "#ef4444"})
    f4.update_traces(marker_size=3)
    f4.add_trace(go.Scatter3d(x=[age], y=[km], z=[price], mode="markers", name="Your car",
                              marker=dict(size=10, color="gold", symbol="diamond")))
    f4.update_layout(template="plotly_dark", height=640, margin=dict(l=0, r=0, t=10, b=0),
                     paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f4, use_container_width=True)
    sm = mkt.groupby("segment").agg(Cars=("selling_price", "size"), Avg_price=("selling_price", "mean"),
                                    Avg_age=("car_age", "mean"), Avg_km=("km_driven", "mean")).round(0)
    st.dataframe(sm.loc[["Budget", "Mid-range", "Premium"]], use_container_width=True)

with tab3:
    pool = mkt[mkt["brand"] == brand]
    pool = pool if len(pool) >= 5 else mkt
    dist = (pool["car_age"] - age).abs() / 5 + (pool["km_driven"] - km).abs() / 50000
    sim = pool.loc[dist.nsmallest(10).index, ["name", "year", "km_driven", "fuel",
                                               "transmission", "owner", "selling_price"]]
    sim = sim.rename(columns={"name": "Car", "km_driven": "Km", "selling_price": "Listed price (₹)"})
    st.caption("The closest real listings in the dataset by brand, age and kilometres.")
    st.dataframe(sim, use_container_width=True, hide_index=True)
    st.metric("Median of these listings", inr(sim["Listed price (₹)"].median()),
              delta=f"{price - sim['Listed price (₹)'].median():+,.0f} vs model estimate")

with tab4:
    res = pd.DataFrame({"Model": ["XGBoost", "Gradient Boosting", "Random Forest",
                                  "Linear Regression", "KNN Regressor"],
                        "Test RMSE (₹)": [200415, 202669, 212571, 214650, 221610]})
    f5 = px.bar(res, x="Test RMSE (₹)", y="Model", orientation="h", text="Test RMSE (₹)",
                title="Test RMSE, lower is better")
    f5.update_layout(template="plotly_dark", height=380, yaxis={"categoryorder": "total descending"},
                     paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f5, use_container_width=True)
    st.write("**Winner: XGBoost**, a boosting ensemble tuned with 5-fold cross-validation and "
             "RandomizedSearchCV. It is about 6.6% better than the Linear Regression baseline. "
             "K-Means (unsupervised) creates the market segments but does not predict prices.")

st.caption("Trained on CarDekho listings (2020 and earlier). Estimates are statistical guidance, "
           "not a valuation or a guarantee.")
