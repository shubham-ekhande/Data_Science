import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocess import add_column_names
from utils.feature_engineering import add_rul, add_features
from model.train_ml import train_model
from model.predict import load_model, predict

# 🔥 FORCE FULL WIDTH
st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 100% !important;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Predictive Maintenance Dashboard")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
">
    <h2 style="margin:0; color:#60a5fa;">
        ✈️ Aircraft Engine Intelligence System
    </h2>
    <p style="margin:5px 0 0 0; color:#9ca3af;">
        Predict Remaining Useful Life (RUL) using AI-powered predictive maintenance
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Dataset (.txt / .csv)", type=["txt", "csv"])

if uploaded_file:

    # Load data
    if uploaded_file.name.endswith(".txt"):
        df = pd.read_csv(uploaded_file, sep=" ", header=None)
        df = df.dropna(axis=1)
    else:
        df = pd.read_csv(uploaded_file)

    # Preprocess
    if "engine_id" not in df.columns:
        df = add_column_names(df)

    if "RUL" not in df.columns:
        df = add_rul(df)

    df = add_features(df)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    X = df.drop(["engine_id", "cycle", "RUL"], axis=1, errors="ignore")
    y = df["RUL"]

    col1, col2 = st.columns(2)
    train_clicked = col1.button("🚀 Train Model")
    predict_clicked = col2.button("🔮 Predict")

    # ======================
    # TRAIN
    # ======================
    if train_clicked:
        model, rmse, mae, r2 = train_model(X, y)

        st.success("Model trained successfully ✅")

        st.subheader("📊 Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", round(rmse, 2))
        m2.metric("MAE", round(mae, 2))
        m3.metric("R² Score", round(r2, 3))

    # ======================
    # PREDICT
    # ======================
    if predict_clicked:

        if not os.path.exists("models/xgb_model.pkl"):
            st.error("Train model first ❌")
        else:
            model = load_model()
            preds = predict(model, X)

            result = pd.DataFrame({"RUL": preds})

            # KPI
            st.subheader("📊 Key Metrics")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg RUL", round(result["RUL"].mean(), 2))
            k2.metric("Min RUL", round(result["RUL"].min(), 2))
            k3.metric("Max RUL", round(result["RUL"].max(), 2))
            k4.metric("Std Dev", round(result["RUL"].std(), 2))

            # Trend
            st.subheader("📈 RUL Trend")
            fig1 = px.line(result.head(500), y="RUL")
            st.plotly_chart(fig1, use_container_width=True)

            # Distribution
            colA, colB = st.columns(2)

            with colA:
                fig2 = px.histogram(result, x="RUL", nbins=30)
                st.plotly_chart(fig2, use_container_width=True)

            def categorize(rul):
                if rul < 20:
                    return "Critical"
                elif rul < 50:
                    return "Warning"
                else:
                    return "Healthy"

            result["Health"] = result["RUL"].apply(categorize)

            with colB:
                fig3 = px.pie(result, names="Health")
                st.plotly_chart(fig3, use_container_width=True)

            # Table
            st.subheader("📋 Predictions")
            st.dataframe(result.head(100), use_container_width=True)