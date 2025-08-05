import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load Models (fake for now)
# ---------------------------
class FakeModel:
    def predict(self, X):
        return np.array([np.random.uniform(-2, 2)])  # random prediction

xgb_model_gold = FakeModel()
mlp_model_gold = FakeModel()
xgb_model_dj = FakeModel()
mlp_model_dj = FakeModel()

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
    else:
        model = xgb_model_gold if model_choice == "XGBoost" else mlp_model_gold

    # Prediction
    predicted_return = model.predict(np.zeros((1, 10)))[0]
    final_capital = capital + (predicted_return * shares)
    profit_or_loss = final_capital - capital

    # Sentiment & Tags
    sentiment_score = round(np.random.uniform(-1, 1), 2)
    technical_tag = "Bullish 📈" if predicted_return > 0 else "Bearish 📉"
    fundamental_tag = "Stable ⚖️" if abs(predicted_return) < 2 else "Volatile 🌪️"

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

    st.markdown("### 💬 Market Signals")
    st.info(f"Sentiment Score: {sentiment_score}")
    st.write(f"📉 Technical Indicator: **{technical_tag}**")
    st.write(f"📊 Fundamental View: **{fundamental_tag}**")

    # Decision
    if predicted_return > 0 and sentiment_score > 0.3:
        st.success("✅ Yes, it's a good time to invest.")
    elif predicted_return > 0:
        st.warning("🟡 Invest with caution. Some risks exist.")
    else:
        st.error("❌ No, it's not a good time to invest.")

    # ---------------------------
    # Chart 1: Historical Price Trend (Selected Asset)
    # ---------------------------
    st.markdown("### 📊 Historical Price Trend")
    dates = pd.date_range("2023-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100  # fake trend
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(dates, prices, label=investment_option, linewidth=2, color="blue")
    ax.set_title(f"{investment_option} Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # Chart 2: Gold vs DJIA Comparison (Side-by-Side)
    # ---------------------------
    st.markdown("### 📊 Gold vs DJIA Comparison")

    # Fake data for both
    gold_prices = np.cumsum(np.random.randn(30)) + 1800  # gold near 1800
    djia_prices = np.cumsum(np.random.randn(30)) + 35000  # DJIA near 35k

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(dates, gold_prices, color="gold", linewidth=2)
        ax1.set_title("Gold Price Trend")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.plot(dates, djia_prices, color="green", linewidth=2)
        ax2.set_title("DJIA Price Trend")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Index Value")
        ax2.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig2)

    st.caption("⚠️ Disclaimer: This is based on model outputs. Markets may change. Invest responsibly.")
