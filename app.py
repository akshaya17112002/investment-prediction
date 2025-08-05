# app.py
import streamlit as st
import numpy as np

# Fake Models (replace later with real models)
class FakeModel:
    def predict(self, X):
        return np.array([np.random.uniform(-2, 2)])

xgb_model_gold = FakeModel()
mlp_model_gold = FakeModel()
xgb_model_dj = FakeModel()
mlp_model_dj = FakeModel()

# --- WEBSITE START ---
st.set_page_config(page_title="Investment Prediction", page_icon="💹", layout="wide")

# Banner
st.image("https://images.unsplash.com/photo-1522202176988-66273c2fd55f", 
         use_column_width=True, caption="💹 AI-Powered Investment Simulator")

st.title("📈 Investment Prediction Simulator")

# Sidebar with images
st.sidebar.header("Choose Investment Option")
st.sidebar.image("https://images.unsplash.com/photo-1611078489935-84b7cbdeb18d", caption="Gold", use_column_width=True)
st.sidebar.image("https://images.unsplash.com/photo-1569025690938-a00729c9eae4", caption="DJIA", use_column_width=True)

# Inputs
capital = st.number_input("💵 Capital ($):", min_value=100.0, value=1000.0, step=100.0)
shares = st.number_input("📦 Shares:", min_value=1, value=10, step=1)
investment_option = st.selectbox("📊 Invest in:", ["DJIA", "Gold"])
model_choice = st.selectbox("🧠 Model:", ["XGBoost", "MLP"])

if st.button("💡 Predict & Explain"):
    # Pick model
    if investment_option == "DJIA":
        model = xgb_model_dj if model_choice == "XGBoost" else mlp_model_dj
    else:
        model = xgb_model_gold if model_choice == "XGBoost" else mlp_model_gold

    predicted_return = model.predict(np.zeros((1, 10)))[0]
    final_capital = capital + (predicted_return * shares)
    profit_or_loss = final_capital - capital

    sentiment_score = round(np.random.uniform(-1, 1), 2)
    technical_tag = "Bullish 📈" if predicted_return > 0 else "Bearish 📉"
    fundamental_tag = "Stable ⚖️" if abs(predicted_return) < 2 else "Volatile 🌪️"

    # --- Output Layout ---
    st.markdown("## 📊 Results")
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

    st.caption("⚠️ Disclaimer: This is based on model outputs. Markets may change. Invest responsibly.")

