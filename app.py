import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AutoWorth - Used Car Fair-Price Bot", page_icon="🚗")

CURRENT_YEAR = 2026
TYPICAL_ERROR = 200_000  # test RMSE of the winning model, in rupees


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

BRANDS = ['Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat',
          'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia',
          'Land', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi',
          'Nissan', 'OpelCorsa', 'Renault', 'Skoda', 'Tata', 'Toyota',
          'Volkswagen', 'Volvo']
DISPLAY = {"Land": "Land Rover", "OpelCorsa": "Opel Corsa"}

st.title("🚗 AutoWorth")
st.subheader("Used Car Fair-Price Bot")
st.write("Tell us about the car and get an estimated fair resale price.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti"),
                         format_func=lambda b: DISPLAY.get(b, b))
    year = st.number_input("Year of manufacture", min_value=1992,
                           max_value=CURRENT_YEAR, value=2018, step=1)
    km_driven = st.number_input("Kilometres driven", min_value=0,
                                max_value=500000, value=50000, step=1000)
with col2:
    fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

with st.expander("More details (optional)"):
    owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner",
                                   "Fourth & Above Owner", "Test Drive Car"])
    seller_type = st.selectbox("Seller type",
                               ["Individual", "Dealer", "Trustmark Dealer"])
    asking = st.number_input("Seller's asking price (Rs), 0 to skip",
                             min_value=0, value=0, step=10000)

if st.button("Estimate fair price", type="primary"):
    row = pd.DataFrame([{
        "km_driven": km_driven,
        "car_age": CURRENT_YEAR - year,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
    }])
    price = float(model.predict(row)[0])
    price = max(price, 20000)  # prices below Rs 20,000 are not realistic

    st.success(f"Estimated fair price: Rs {price:,.0f}")
    low, high = max(price - TYPICAL_ERROR, 0), price + TYPICAL_ERROR
    st.caption(f"Typical range: Rs {low:,.0f} to Rs {high:,.0f} "
               "(the model's typical error is about Rs 2 lakh).")

    if asking > 0:
        if asking > high:
            st.warning("The asking price looks HIGH compared with similar cars.")
        elif asking < low:
            st.info("The asking price looks LOW. Check the car's condition carefully.")
        else:
            st.success("The asking price looks FAIR.")

st.divider()
st.caption("Model: XGBoost trained on CarDekho listings. Estimates are statistical "
           "guidance, not a valuation.")
