import matplotlib.dates as mdates

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

# FIX: Clean x-axis formatting
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.xticks(rotation=45)

st.pyplot(fig)

# ---------------------------
# EXPLANATION (Better Storytelling)
# ---------------------------
st.markdown("### 📖 Why This Prediction?")

explanation = []

# Sentiment reasoning
if sentiment_score > 0.3:
    explanation.append("💚 Market sentiment is positive, indicating investor confidence.")
elif sentiment_score < -0.3:
    explanation.append("💔 Market sentiment is negative, showing lack of confidence.")
else:
    explanation.append("😐 Market sentiment is neutral, with no strong bias.")

# Technical reasoning
if "Bullish" in technical_tag:
    explanation.append("📈 Technical signals (like moving averages) are trending upward.")
else:
    explanation.append("📉 Technical signals (like moving averages) are trending downward.")

# Fundamental reasoning
if "Stable" in fundamental_tag:
    explanation.append("⚖️ Fundamentals remain stable, with no major economic shocks.")
else:
    explanation.append("🌪️ Fundamentals are volatile, indicating macro uncertainty.")

# Final decision logic
if predicted_return > 0 and sentiment_score > 0.3:
    final_reason = "✅ Overall, multiple signals are aligned positively — suggesting it's a good time to invest."
elif predicted_return > 0:
    final_reason = "🟡 Some signals are positive but not all — caution is advised."
else:
    final_reason = "❌ Most signals indicate weakness — not a good time to invest."

# Display explanation
for line in explanation:
    st.write(line)
st.subheader(final_reason)
