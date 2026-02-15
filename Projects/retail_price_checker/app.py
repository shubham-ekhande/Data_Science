import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load Model & Feature Columns
# -------------------------------
model = joblib.load("models/demand_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("💰 Smart Pricing Optimization System")

st.write("Adjust inputs and find the optimal revenue-maximizing price.")

# -------------------------------
# User Inputs
# -------------------------------
unit_price = st.number_input("Current Price", value=150.0)
freight_price = st.number_input("Freight Price", value=20.0)
customers = st.number_input("Number of Customers", value=100)
holiday = st.selectbox("Holiday", [0, 1])
month = st.slider("Month", 1, 12, 6)

# -------------------------------
# Create Input DataFrame
# -------------------------------
def create_input_dataframe():
    
    # Create empty dataframe with training columns
    input_df = pd.DataFrame(columns=feature_columns)
    input_df.loc[0] = 0  # initialize all columns to 0
    
    # Helper function
    def set_value(col, value):
        if col in input_df.columns:
            input_df.at[0, col] = value

    # Set known values
    set_value("unit_price", unit_price)
    set_value("freight_price", freight_price)
    set_value("product_score", 4.0)
    set_value("customers", customers)
    set_value("weekday", 1)
    set_value("weekend", 0)
    set_value("holiday", holiday)
    set_value("volume", 2000)
    set_value("comp_1", unit_price * 0.95)
    set_value("comp_2", unit_price * 1.05)
    set_value("comp_3", unit_price * 1.02)
    set_value("lag_price", unit_price)
    set_value("month", month)

    return input_df


# -------------------------------
# Optimization Logic
# -------------------------------
if st.button("Find Optimal Price"):

    input_data = create_input_dataframe()

    price_range = np.linspace(unit_price * 0.5, unit_price * 1.5, 50)
    revenues = []

    for price in price_range:
        input_data["unit_price"] = price
        predicted_qty = model.predict(input_data)[0]
        revenue = price * predicted_qty
        revenues.append(revenue)

    optimal_index = np.argmax(revenues)
    optimal_price = price_range[optimal_index]
    max_revenue = revenues[optimal_index]

    st.success(f"📌 Current Price: {unit_price:.2f}")
    st.success(f"🚀 Optimal Price: {optimal_price:.2f}")
    st.success(f"💵 Maximum Revenue: {max_revenue:.2f}")

    # -------------------------------
    # Plot Revenue Curve
    # -------------------------------
    fig, ax = plt.subplots()
    ax.plot(price_range, revenues)
    ax.axvline(optimal_price, linestyle="--")
    ax.set_xlabel("Price")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue vs Price")

    st.pyplot(fig)
