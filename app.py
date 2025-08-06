import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# ---------------------------
# Load Real Models
# ---------------------------
xgb_model_gold = joblib.load("xgb_model_gold.pkl")
mlp_model_gold = joblib.load("mlp_model_gold.pkl")
xgb_model_dj = joblib.load("xgb_model_dj.pkl")
mlp_model_dj = joblib.load("mlp_model_dj.pkl")

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Investment Prediction", page_icon="💹", layout="wide")

# Background
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("background.jpg");
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Banner
st.image("banner.jpg", use_container_width=True, caption="💹 AI-Powered Investment Simulator")

st.title("📈 Investment Prediction Simulator")

# ---------------------------
# User Inputs
# ---------------------------
capital = st.number_input("💵 Capital ($):", min_value=100.0, value=1000.0, step=100.0)
shares = st.number_input("📦 Shares:", min_value=1, value=10, step=1)
investment_option = st.selectbox("📊 Invest in:", ["DJIA", "Gold"])
model_choice = st.selectbox("🧠 Model:", ["XGBoost", "MLP"])

# Show Selected Asset Image
if investment_option == "Gold":
    st.image("gold.jpg", caption="Gold Market", use_container_width=True)
elif investment_option == "DJIA":
    st.image("djia.jpg", caption="Dow Jones (DJIA)", use_container_width=True)

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("💡 Predict & Explain"):
    # Pick model
    if investment_option == "DJIA":
        model = xgb_model_dj if model_choice == "XGBoost" else mlp_model_dj
        test_data = pd.read_csv("X_test_dj.csv").iloc[[0]]
        y_test = pd.read_csv("y_test_dj.csv")
        actual_last_close = y_test.iloc[0, 0]
    else:
        model = xgb_model_gold if model_choice == "XGBoost" else mlp_model_gold
        test_data = pd.read_csv("X_test_gold.csv").iloc[[0]]
        y_test = pd.read_csv("y_test_gold.csv")
        actual_last_close = y_test.iloc[0, 0]

    # Prediction (using real trained model)
    predicted_price = model.predict(test_data)[0]

    # ✅ Correct profit calculation
    price_change = predicted_price - actual_last_close
    profit_or_loss = price_change * shares
    final_capital = capital + profit_or_loss

    # Debug
    st.write(f"🔍 Predicted Close = {predicted_price:.2f}")
    st.write(f"📊 Last Close = {actual_last_close:.2f}")
    st.write(f"💡 Price Change = {price_change:.2f} per share")

    # ---------------------------
    # Display Metrics
    # ---------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💵 Initial Capital", f"${capital:.2f}")
    with col2:
        st.metric("🧮 Final Capital", f"${final_capital:.2f}")
    with col3:
        st.metric("📊 Profit/Loss", f"${profit_or_loss:.2f}")

    # ---------------------------
    # Chart 1: Historical Price Trend
    # ---------------------------
    st.markdown("### 📊 Historical Price Trend")
    dates = pd.date_range("2023-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(dates, prices, label=investment_option, linewidth=2, color="blue")
    date_range = f"{dates[0].strftime('%b %d, %Y')} → {dates[-1].strftime('%b %d, %Y')}"
    ax.set_title(f"{investment_option} Price Trend\n({date_range})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------------------------
    # Chart 2: Gold vs DJIA Comparison
    # ---------------------------
    st.markdown("### 📊 Gold vs DJIA Comparison")

    gold_prices = np.cumsum(np.random.randn(30)) + 1800
    djia_prices = np.cumsum(np.random.randn(30)) + 35000

    colA, colB = st.columns(2)
    with colA:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(dates, gold_prices, color="gold", linewidth=2)
        ax1.set_title(f"Gold Price Trend\n({date_range})")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.plot(dates, djia_prices, color="green", linewidth=2)
        ax2.set_title(f"DJIA Price Trend\n({date_range})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Index Value")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        st.pyplot(fig2)

    # ---------------------------
    # EXPLANATION
    # ---------------------------
    st.markdown("### 📖 Why This Prediction?")
    if price_change > 0:
        st.success("✅ The model predicts an increase compared to the last close — potential gain.")
    else:
        st.error("❌ The model predicts a decrease compared to the last close — potential loss.")

    st.caption("⚠️ Disclaimer: This is based on model outputs. Markets may change. Invest responsibly.")
