"""
Throwaway smoke test for AuctionNetGymEnv.
Run from project root: python test_env.py
Delete after confirming everything works.
"""

import sys

def step(msg):
    print(f"  [ ] {msg}", end="", flush=True)

def ok():
    print("\r  [x]", flush=True)

def fail(e):
    print(f"\r  [!] FAILED — {e}")
    sys.exit(1)


print("=" * 55)
print("  AuctionNetGymEnv Smoke Test")
print("=" * 55)

# ── 1. Config ────────────────────────────────────────────
print("\n[1] Loading config")
try:
    step("Reading configs/default.yaml")
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)["environment"]
    # Override pv_num for smoke test — 1K PVs instead of 500K
    config["pv_num"] = 1_000
    ok()
    print(f"      budget={config['budget']}, num_ticks={config['num_ticks']}, "
          f"player_index={config['player_index']}, pv_num={config['pv_num']} (smoke test override)")
except Exception as e:
    fail(e)

# ── 2. AuctionNet import ─────────────────────────────────
print("\n[2] Accessing AuctionNet modules")
try:
    import sys, os
    auctionnet_abs = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(".")), "AuctionNet")
    )
    if auctionnet_abs not in sys.path:
        sys.path.insert(0, auctionnet_abs)

    step("Importing BiddingEnv")
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    ok()

    step("Importing NeurIPSPvGen")
    from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen
    ok()

    step("Importing PidBiddingStrategy")
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    ok()

    step("Importing PidBiddingStrategy (sole competitor — no model files needed)")
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
    ok()
except Exception as e:
    fail(e)

# ── 3. Environment init ──────────────────────────────────
print("\n[3] Initialising AuctionNetGymEnv")
try:
    step("Importing wrapper")
    from environment.gym_wrapper import AuctionNetGymEnv
    ok()

    step("Instantiating env (builds BiddingEnv, PvGen, 47 competitors)")
    env = AuctionNetGymEnv(config)
    ok()

    step("Checking observation_space shape")
    assert env.observation_space.shape == (7,), env.observation_space.shape
    ok()

    step("Checking action_space shape")
    assert env.action_space.shape == (1,), env.action_space.shape
    ok()
except Exception as e:
    fail(e)

# ── 4. SB3 compliance ────────────────────────────────────
print("\n[4] Running SB3 check_env")
try:
    step("check_env (verifies Gymnasium API contract)")
    from stable_baselines3.common.env_checker import check_env
    check_env(env)
    ok()
except Exception as e:
    fail(e)

# ── 5. Reset ─────────────────────────────────────────────
print("\n[5] Resetting environment")
try:
    step("env.reset()")
    obs, info = env.reset()
    ok()
    print(f"      obs shape : {obs.shape}")
    print(f"      obs values: {obs}")
except Exception as e:
    fail(e)

# ── 6. Episode rollout ───────────────────────────────────
print("\n[6] Running one full episode (random actions)")
print(f"  {'tick':>4}  {'won':>5}  {'cost':>8}  {'conv':>8}  {'reward':>8}  {'budget':>8}")
print(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
try:
    total_reward = 0
    for tick in range(48):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  {tick+1:>4}  "
            f"{info['won']:>5}  "
            f"{info['cost']:>8.2f}  "
            f"{info['conversion_value']:>8.4f}  "
            f"{reward:>8.4f}  "
            f"{info['remaining_budget']:>8.2f}"
        )
        if terminated:
            print(f"\n  Budget exhausted — episode ended at tick {tick+1}")
            break
except Exception as e:
    fail(e)

# ── Summary ──────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  Total reward : {total_reward:.4f}")
print(f"  Final obs    : {obs}")
print("  All checks passed.")
print("=" * 55)
