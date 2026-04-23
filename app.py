import streamlit as st
import numpy as np
from stable_baselines3 import PPO
import random

st.set_page_config(page_title="AI Bidding Game", page_icon="🎮", layout="centered")

st.title("🎮 Can the AI Beat the Market?")
st.markdown("Play with the budget and time of day to see how the Deep RL agent adjusts its strategy to win live ad auctions!")

# Load the model
@st.cache_resource
def load_model():
    try:
        return PPO.load("saved_models/ppo_baseline_best/best_model.zip")
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("⚠️ Model not found! Ensure 'saved_models/ppo_baseline_best/best_model.zip' exists.")
    st.stop()

# --- Simple UI ---
st.markdown("### 1. Set the Scenario")
col1, col2, col3 = st.columns(3)
with col1:
    budget_pct = st.slider("💰 Budget Remaining (%)", 0, 100, 50)
with col2:
    time_pct = st.slider("⏰ Time of Day (%)", 0, 100, 50, help="0% is Morning, 100% is Midnight")
with col3:
    p_value_pct = st.slider("👤 User Buy Probability (%)", 0, 100, 80, help="How likely is this specific user to buy the product?")

# --- Advanced UI ---
with st.expander("⚙️ Advanced Market Parameters (Optional)"):
    st.markdown("The AI uses these to calculate the precise state of the market.")
    c1, c2 = st.columns(2)
    mean_pval = c1.slider("Opportunity Quality (Mean)", 0.0, 1.0, 0.05)
    std_pval = c2.slider("Opportunity Diversity (Std)", 0.0, 1.0, 0.01)
    norm_price = c1.slider("Market Price Proxy", 0.0, 0.1, 0.02)
    recent_win_rate = c2.slider("Recent Win Rate", 0.0, 1.0, 0.1)
    pacing_ratio = st.slider("Spend Pacing (>1 = Overspending)", 0.0, 3.0, 1.0)

# Build Observation
obs = np.array([
    budget_pct / 100.0,
    time_pct / 100.0,
    mean_pval,
    std_pval,
    norm_price,
    recent_win_rate,
    pacing_ratio
], dtype=np.float32)

# Get AI Prediction
action, _ = model.predict(obs, deterministic=True)
action_clipped = np.clip(action.item() if getattr(action, "size", 0) == 1 else action, -1.0, 1.0)
bid_multiplier = ((action_clipped + 1.0) / 2.0) * 150.0

st.markdown("---")
st.markdown("### 2. Live Ad Auction Simulator")

# Generate realistic context
random.seed(int(budget_pct + time_pct + p_value_pct)) 
competitor_names = ["Amazon", "Nike", "Target", "Best Buy", "AdTech Corp", "Zappos", "Walmart"]
user_personas = [
    "Sarah (28) - Browsing running shoes",
    "John (45) - Reading tech blogs",
    "Emily (32) - Watching a cooking video",
    "Michael (19) - Scrolling social media",
    "Jessica (50) - Looking at vacation packages"
]
competitor_name = random.choice(competitor_names)
user_persona = random.choice(user_personas)

st.markdown(f"**👀 {user_persona}** just opened a webpage! Based on our cookie tracking, they have a **{p_value_pct}% chance** of buying our product.")

p_value = p_value_pct / 100.0

# Competitor logic: Competitors have CPA targets around $100.
# So their baseline bid is roughly 100 * p_value. We add +/- $15 randomness to simulate live market fluctuations.
competitor_baseline = 100.0 * p_value
competitor_bid = max(0.01, round(random.uniform(competitor_baseline - 15.0, competitor_baseline + 15.0), 2))

ai_bid = round(bid_multiplier * p_value, 2)

c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"🤖 **Our AI Agent Bids**\n\n# ${ai_bid:.2f}")
with c2:
    st.warning(f"🏢 **{competitor_name} Bids**\n\n# ${competitor_bid:.2f}")
with c3:
    if ai_bid > competitor_bid:
        st.success(f"🎉 **AI WINS!**\n\nWe secured the ad space! We pay the market clearing price of **${competitor_bid:.2f}** to show our ad to {user_persona.split(' ')[0]}.")
    else:
        st.error(f"❌ **{competitor_name.upper()} WINS**\n\nThe AI decided this user was too expensive, refused to overpay, and let {competitor_name} take the ad space.")

st.markdown("---")
st.markdown(f"**Behind the Scenes:** The neural network looked at the remaining budget and time, and decided the mathematically optimal strategy was to use a **{bid_multiplier:.1f}x multiplier** on the user's value.")
