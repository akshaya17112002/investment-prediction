import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    # Pick model & load test data
    if investment_option == "DJIA":
        model = xgb_model_dj if model_choice == "XGBoost" else mlp_model_dj
        test_data = pd.read_csv("X_test_dj.csv").iloc[[0]]  # first row DJIA test
    else:
        model = xgb_model_gold if model_choice == "XGBoost" else mlp_model_gold
        test_data = pd.read_csv("X_test_gold.csv").iloc[[0]]  # first row Gold test

    # Debug: show model type
    st.write(f"🔍 Using model type: {type(model)}")

    # Prediction
    predicted_return = model.predict(test_data)[0]
    final_capital = capital + (predicted_return * shares)
    profit_or_loss = final_capital - capital

    # Sentiment & Tags (simulated)
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

    # ---------------------------
    # Chart 1: Historical Price Trend (real data)
    # ---------------------------
    st.markdown("### 📊 Historical Price Trend")

    if investment_option == "Gold":
        gold_df = pd.read_csv("gold_prices.csv", index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(gold_df.index, gold_df["Close"], color="gold", linewidth=2, label="Gold")
        ax.set_title("Gold Price Trend (2015 → 2024)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (USD)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    elif investment_option == "DJIA":
        djia_df = pd.read_csv("djia_prices.csv", index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(djia_df.index, djia_df["Close"], color="green", linewidth=2, label="DJIA")
        ax.set_title("DJIA Price Trend (2015 → 2024)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Index Value")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    # ---------------------------
    # Chart 2: Gold vs DJIA Comparison (real data)
    # ---------------------------
    st.markdown("### 📊 Gold vs DJIA Comparison")

    gold_df = pd.read_csv("gold_prices.csv", index_col=0, parse_dates=True)
    djia_df = pd.read_csv("djia_prices.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(gold_df.index, gold_df["Close"], color="gold", linewidth=2, label="Gold")
    ax.plot(djia_df.index, djia_df["Close"], color="green", linewidth=2, label="DJIA")
    ax.set_title("Gold vs DJIA (2015 → 2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # EXPLANATION
    # ---------------------------
    st.markdown("### 📖 Why This Prediction?")
    explanation = []

    if sentiment_score > 0.3:
        explanation.append("💚 Market sentiment is positive, indicating investor confidence.")
    elif sentiment_score < -0.3:
        explanation.append("💔 Market sentiment is negative, showing lack of confidence.")
    else:
        explanation.append("😐 Market sentiment is neutral, with no strong bias.")

    if "Bullish" in technical_tag:
        explanation.append("📈 Technical signals (like moving averages) are trending upward.")
    else:
        explanation.append("📉 Technical signals (like moving averages) are trending downward.")

    if "Stable" in fundamental_tag:
        explanation.append("⚖️ Fundamentals remain stable, with no major economic shocks.")
    else:
        explanation.append("🌪️ Fundamentals are volatile, indicating macro uncertainty.")

    if predicted_return > 0 and sentiment_score > 0.3:
        final_reason = "✅ Overall, multiple signals are aligned positively — suggesting it's a good time to invest."
    elif predicted_return > 0:
        final_reason = "🟡 Some signals are positive but not all — caution is advised."
    else:
        final_reason = "❌ Most signals indicate weakness — not a good time to invest."

    for line in explanation:
        st.write(line)
    st.subheader(final_reason)

    st.caption("⚠️ Disclaimer: This is based on model outputs. Markets may change. Invest responsibly.")
