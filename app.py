import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load Models (currently using FakeModel → replace with real ones later)
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

# Apply Custom Background Image (CSS hack)
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

# Banner Image
st.image("banner.jpg", use_column_width=True, caption="💹 AI-Powered Investment Simulator")

st.title("📈 Investment Prediction Simulator")

# Sidebar with Gold & DJIA Images
st.sidebar.header("Choose Investment Option")
st.sidebar.image("gold.jpg", caption="Gold", use_column_width=True)
st.sidebar.image("djia.jpg", caption="DJIA", use_column_width=True)

# ---------------------------
# User Inputs
# ---------------------------
capital = st.number_input("💵 Capital ($):", min_value=100.0, value=1000.0, step=100.0)
shares = st.number_input("📦 Shares:", min_value=1, value=10, step=1)
investment_option = st.selectbox("📊 Invest in:", ["DJIA", "Gold"])
model_choice = st.selectbox("🧠 Model:", ["XGBoost", "MLP"])

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
    # Sample Chart (Price Trends)
    # ---------------------------
    st.markdown("### 📊 Historical Price Trend")
    dates = pd.date_range("2023-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100  # fake trend
    fig, ax = plt.subplots()
    ax.plot(dates, prices, label=investment_option, linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.caption("⚠️ Disclaimer: This is based on model outputs. Markets may change. Invest responsibly.")
