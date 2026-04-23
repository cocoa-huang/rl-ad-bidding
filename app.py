import streamlit as st
import numpy as np
from stable_baselines3 import PPO
import random
import time

st.set_page_config(page_title="AI Bidding Game", page_icon="🎮", layout="centered")

st.title("🎮 Can the AI Beat the Market?")
st.markdown("Play with the budget and time of day to see how the Deep RL agent adjusts its strategy to win live ad auctions!")

# Lightweight styling to make the demo feel more "app-like"
st.markdown(
    """
<style>
  .card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px 14px;
    background: rgba(255,255,255,0.03);
  }
  .card h4 { margin: 0 0 6px 0; font-size: 0.95rem; opacity: 0.95; }
  .card .big { font-size: 1.6rem; font-weight: 750; line-height: 1.1; margin-top: 2px; }
  .muted { opacity: 0.75; }
</style>
""",
    unsafe_allow_html=True,
)

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

st.markdown("### Live Auction Playground")
st.caption("Tune the market + user intent on the left. Watch the live auction outcome on the right.")

left, right = st.columns([1.15, 1.0], vertical_alignment="top")
with left:
    st.markdown("#### Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_pct = st.slider("💰 Budget Remaining (%)", 0, 100, 50)
    with col2:
        time_pct = st.slider("⏰ Time of Day (%)", 0, 100, 50, help="0% is Morning, 100% is Midnight")
    with col3:
        p_value_pct = st.slider(
            "👤 User Buy Probability (%)",
            0,
            100,
            80,
            help="How likely is this specific user to buy the product?",
        )

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

with right:
    st.markdown("#### Live Auction Result")
    st.markdown(
        f"**👀 {user_persona}** opened a page. Cookie intent says **{p_value_pct}%** chance they buy."
    )

    card1, card2 = st.columns(2)
    with card1:
        st.markdown(
            f"""
<div class="card">
  <h4>🤖 Our AI bid</h4>
  <div class="big">${ai_bid:.2f}</div>
  <div class="muted">multiplier: {bid_multiplier:.1f}×</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with card2:
        st.markdown(
            f"""
<div class="card">
  <h4>🏢 {competitor_name} bid</h4>
  <div class="big">${competitor_bid:.2f}</div>
  <div class="muted">~ CPA $100 × p(user)</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if ai_bid > competitor_bid:
        st.success(
            f"🎉 **We win the slot.** Clearing price **${competitor_bid:.2f}**. Our ad shows to **{user_persona.split(' ')[0]}**."
        )
    else:
        st.error(
            f"❌ **We lose the slot.** {competitor_name} shows their ad. (AI refused to overpay.)"
        )

st.markdown(f"**Behind the Scenes:** The neural network looked at the remaining budget and time, and decided the mathematically optimal strategy was to use a **{bid_multiplier:.1f}x multiplier** on the user's value.")

st.markdown("---")
st.markdown("### 3. 5-User Browser Wall (Scroll + Pop-up Ads)")
st.caption(
    "Each mini screen is a simulated user session scrolling a feed. When they hit an ad slot, either our ad or a competitor’s ad wins the auction."
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _auction_result(*, p_value: float, bid_multiplier: float, rng: random.Random) -> dict:
    competitor_names = ["Amazon", "Nike", "Target", "Best Buy", "AdTech Corp", "Zappos", "Walmart"]
    competitor_name = rng.choice(competitor_names)

    competitor_baseline = 100.0 * p_value
    competitor_bid = max(0.01, round(rng.uniform(competitor_baseline - 15.0, competitor_baseline + 15.0), 2))
    ai_bid = round(bid_multiplier * p_value, 2)

    ai_wins = ai_bid > competitor_bid
    clearing_price = competitor_bid if ai_wins else ai_bid

    return {
        "competitor_name": competitor_name,
        "competitor_bid": competitor_bid,
        "ai_bid": ai_bid,
        "ai_wins": ai_wins,
        "clearing_price": clearing_price,
    }


def _init_wall_state(seed: int) -> None:
    rng = random.Random(seed)
    user_personas = [
        "Sarah (28) - Browsing running shoes",
        "John (45) - Reading tech blogs",
        "Emily (32) - Watching a cooking video",
        "Michael (19) - Scrolling social media",
        "Jessica (50) - Looking at vacation packages",
    ]

    sessions = []
    for i in range(5):
        session_rng_seed = rng.randint(0, 10_000_000)
        session_rng = random.Random(session_rng_seed)

        persona = session_rng.choice(user_personas)
        # Per-user intent fluctuates around the global slider.
        base_p = p_value_pct / 100.0
        p_user = _clamp01(session_rng.normalvariate(base_p, 0.12))

        # Feed design: fixed length with an ad slot at a predictable spot.
        feed_len = session_rng.randint(18, 26)
        ad_slot_idx = session_rng.randint(7, 14)
        scroll_speed = session_rng.choice([1, 1, 1, 2, 2, 3])  # skew slower

        sessions.append(
            {
                "id": i,
                "persona": persona,
                "p_user": p_user,
                "feed_len": feed_len,
                "ad_slot_idx": ad_slot_idx,
                "pos": 0,
                "scroll_speed": scroll_speed,
                "rng_seed": session_rng_seed,
                "last_event": None,
                "last_event_tick": None,
            }
        )

    st.session_state.wall = {
        "seed": seed,
        "tick": 0,
        "sessions": sessions,
        "auto": False,
        "last_step_ms": 250,
    }


def _step_wall() -> None:
    wall = st.session_state.wall
    wall["tick"] += 1
    tick = wall["tick"]

    for s in wall["sessions"]:
        rng = random.Random(s["rng_seed"] + tick * 97)

        # Scroll forward with a bit of jitter.
        delta = s["scroll_speed"] + rng.choice([0, 0, 1, -1])
        delta = max(0, delta)
        s["pos"] = int(min(s["feed_len"] - 1, s["pos"] + delta))

        # When the user hits the ad slot, run an auction and create a "popup".
        if s["pos"] == s["ad_slot_idx"]:
            res = _auction_result(p_value=s["p_user"], bid_multiplier=bid_multiplier, rng=rng)
            if res["ai_wins"]:
                headline = "Our Ad Wins"
                sub = f"Paid ${res['clearing_price']:.2f} (AI bid ${res['ai_bid']:.2f} vs {res['competitor_name']} ${res['competitor_bid']:.2f})"
                tone = "success"
            else:
                headline = f"{res['competitor_name']} Wins"
                sub = f"AI bid ${res['ai_bid']:.2f} vs {res['competitor_name']} ${res['competitor_bid']:.2f}"
                tone = "error"

            s["last_event"] = {"headline": headline, "sub": sub, "tone": tone}
            s["last_event_tick"] = tick

        # When they reach the bottom, "reload" a new page/feed.
        if s["pos"] >= s["feed_len"] - 1:
            s["pos"] = 0
            s["ad_slot_idx"] = rng.randint(7, 14)
            s["feed_len"] = rng.randint(18, 26)
            s["scroll_speed"] = rng.choice([1, 1, 1, 2, 2, 3])
            s["p_user"] = _clamp01(rng.normalvariate(p_value_pct / 100.0, 0.12))
            s["last_event"] = {"headline": "New Page Load", "sub": "Fresh content + new auction slot", "tone": "info"}
            s["last_event_tick"] = tick


wall_seed = int(budget_pct + time_pct * 10 + p_value_pct * 100)
if "wall" not in st.session_state:
    _init_wall_state(wall_seed)
elif st.session_state.wall.get("seed") != wall_seed:
    _init_wall_state(wall_seed)

controls = st.columns([1, 1, 2, 2])
with controls[0]:
    if st.button("▶️ Step", use_container_width=True):
        _step_wall()
        st.rerun()
with controls[1]:
    if st.button("🔄 Reset", use_container_width=True):
        _init_wall_state(wall_seed)
        st.rerun()
with controls[2]:
    st.session_state.wall["auto"] = st.toggle("Auto-run", value=st.session_state.wall["auto"])
with controls[3]:
    st.session_state.wall["last_step_ms"] = st.slider(
        "Auto speed (ms / step)", 100, 1200, int(st.session_state.wall["last_step_ms"]), 50
    )

cols = st.columns(5)
tick = st.session_state.wall["tick"]
for col, s in zip(cols, st.session_state.wall["sessions"]):
    with col:
        st.markdown("#### 🖥️")
        st.markdown(f"**{s['persona'].split(' - ')[0]}**")
        st.caption(f"Intent (p): **{int(round(s['p_user'] * 100))}%** · Tick: **{tick}**")

        # Scroll indicator
        st.progress((s["pos"] + 1) / max(1, s["feed_len"]))
        st.caption(f"Scroll: item **{s['pos'] + 1} / {s['feed_len']}** · Ad slot at **{s['ad_slot_idx'] + 1}**")

        # Mini "feed"
        start = max(0, s["pos"] - 2)
        end = min(s["feed_len"], s["pos"] + 3)
        for idx in range(start, end):
            if idx == s["ad_slot_idx"]:
                if idx == s["pos"]:
                    st.markdown("**🟨 [AD SLOT]** (in view)")
                else:
                    st.markdown("🟨 [ad slot]")
            else:
                marker = "➡️ " if idx == s["pos"] else ""
                st.markdown(f"{marker}Content card {idx + 1}")

        # Popup panel for last events (fade after a few ticks).
        ev = s.get("last_event")
        if ev and s.get("last_event_tick") is not None and tick - s["last_event_tick"] <= 6:
            if ev["tone"] == "success":
                st.success(f"**{ev['headline']}**\n\n{ev['sub']}")
            elif ev["tone"] == "error":
                st.error(f"**{ev['headline']}**\n\n{ev['sub']}")
            else:
                st.info(f"**{ev['headline']}**\n\n{ev['sub']}")

if st.session_state.wall["auto"]:
    _step_wall()
    time.sleep(st.session_state.wall["last_step_ms"] / 1000.0)
    st.rerun()
