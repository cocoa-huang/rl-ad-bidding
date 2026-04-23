"""
Smoke test for PPOAgent.
Run from project root: conda run -n rl-ad-bidding python test_agent.py

Validates the agent in isolation — no train.py needed. Covers:
  1. Initialization with a plain env
  2. select_action() output shape and dtype
  3. Short training run (one full rollout buffer + gradient update)
  4. save() / load() round-trip
  5. evaluate() returns correct RTB metrics
  6. VecNormalize save/load round-trip
"""

import shutil
import sys


def step(msg):
    print(f"  [ ] {msg}", end="", flush=True)

def ok():
    print("\r  [x]", flush=True)

def fail(e):
    print(f"\r  [!] FAILED — {e}")
    sys.exit(1)


print("=" * 55)
print("  PPOAgent Smoke Test")
print("=" * 55)

# ── 1. Config ────────────────────────────────────────────
print("\n[1] Loading config")
try:
    step("Reading configs/default.yaml")
    import yaml
    with open("configs/default.yaml") as f:
        full_cfg = yaml.safe_load(f)
    env_cfg = full_cfg["environment"].copy()
    agent_cfg = full_cfg["agent"].copy()
    # Small pv_num so each tick runs fast
    env_cfg["pv_num"] = 1000
    # One rollout buffer = n_steps steps. With a single env that means 384 steps
    # = 8 complete 48-tick episodes. Use 400 to guarantee at least one update.
    agent_cfg["total_timesteps"] = 400
    ok()
    print(f"      n_steps={agent_cfg.get('n_steps', 384)}, "
          f"total_timesteps={agent_cfg['total_timesteps']} (smoke test override)")
except Exception as e:
    fail(e)

# ── 2. Imports ───────────────────────────────────────────
print("\n[2] Importing modules")
try:
    step("AuctionNetGymEnv")
    from environment.gym_wrapper import AuctionNetGymEnv
    ok()

    step("PPOAgent")
    from agents.ppo_agent import PPOAgent
    ok()

    step("SB3 VecEnv utilities")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    ok()
except Exception as e:
    fail(e)

# ── 3. Init ──────────────────────────────────────────────
print("\n[3] Initialising PPOAgent with plain env")
try:
    step("Building DummyVecEnv")
    vec_env = DummyVecEnv([lambda: AuctionNetGymEnv(env_cfg)])
    ok()

    step("Instantiating PPOAgent")
    agent = PPOAgent(
        vec_env.observation_space,
        vec_env.action_space,
        agent_cfg,
        env=vec_env,
    )
    ok()

    step("Checking n_steps == 384")
    assert agent.model.n_steps == agent_cfg.get("n_steps", 384), \
        f"expected 384, got {agent.model.n_steps}"
    ok()

    step("Checking ent_coef == 0.01")
    assert abs(agent.model.ent_coef - agent_cfg.get("ent_coef", 0.01)) < 1e-6, \
        f"expected 0.01, got {agent.model.ent_coef}"
    ok()
except Exception as e:
    fail(e)

# ── 4. select_action() ───────────────────────────────────
print("\n[4] Testing select_action()")
try:
    import numpy as np

    step("Reset env, get first obs")
    obs = vec_env.reset()
    ok()

    step("select_action(obs, deterministic=False) — stochastic")
    action, lp, val = agent.select_action(obs, deterministic=False)
    assert action.shape == (1, 1), f"expected (1,1), got {action.shape}"
    assert action.dtype in (np.float32, np.float64), f"unexpected dtype {action.dtype}"
    assert lp is None and val is None
    ok()
    print(f"      action = {action.flatten()[0]:.4f}  (in [-1, 1]; rescaled to [0, {env_cfg['max_bid_multiplier']}] in step())")

    step("select_action(obs, deterministic=True) — greedy")
    action_det, _, _ = agent.select_action(obs, deterministic=True)
    assert action_det.shape == (1, 1)
    ok()
except Exception as e:
    fail(e)

# ── 5. Short training run ────────────────────────────────
print("\n[5] Short training run (one rollout buffer + gradient update)")
try:
    step(f"agent.update(total_timesteps={agent_cfg['total_timesteps']})")
    metrics = agent.update(total_timesteps=agent_cfg["total_timesteps"])
    assert metrics["trained_timesteps"] == agent_cfg["total_timesteps"]
    ok()
    print(f"      trained_timesteps = {metrics['trained_timesteps']}")
except Exception as e:
    fail(e)

# ── 6. save() / load() round-trip (plain env) ────────────
print("\n[6] save() / load() round-trip — plain env")
_SAVE_PATH = "saved_models/_smoke_test_plain"
try:
    step("save()")
    agent.save(_SAVE_PATH)
    from pathlib import Path
    assert Path(_SAVE_PATH + ".zip").exists(), "model .zip not found"
    assert not Path(_SAVE_PATH + "_vecnormalize.pkl").exists(), \
        "unexpected vecnorm sidecar for plain env"
    ok()

    step("load() and re-attach env")
    agent2 = PPOAgent(
        vec_env.observation_space,
        vec_env.action_space,
        agent_cfg,
    )
    agent2.load(_SAVE_PATH, env=vec_env)
    assert agent2.model is not None
    ok()

    step("select_action() after load")
    obs = vec_env.reset()
    action_loaded, _, _ = agent2.select_action(obs, deterministic=True)
    assert action_loaded.shape == (1, 1)
    ok()
except Exception as e:
    fail(e)

# ── 7. VecNormalize save/load round-trip ─────────────────
print("\n[7] save() / load() round-trip — VecNormalize env")
_SAVE_PATH_VN = "saved_models/_smoke_test_vecnorm"
try:
    step("Build VecNormalize-wrapped env")
    raw_env = DummyVecEnv([lambda: AuctionNetGymEnv(env_cfg)])
    norm_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True)
    ok()

    step("Train briefly to accumulate normalizer stats")
    agent_vn = PPOAgent(
        norm_env.observation_space,
        norm_env.action_space,
        agent_cfg,
        env=norm_env,
    )
    agent_vn.update(total_timesteps=agent_cfg["total_timesteps"])
    ok()

    step("save() — expect model .zip + vecnormalize .pkl")
    agent_vn.save(_SAVE_PATH_VN)
    assert Path(_SAVE_PATH_VN + ".zip").exists(), "model .zip not found"
    assert Path(_SAVE_PATH_VN + "_vecnormalize.pkl").exists(), \
        "vecnormalize .pkl not found"
    ok()

    step("load() — stats restored into fresh VecNormalize env")
    fresh_raw = DummyVecEnv([lambda: AuctionNetGymEnv(env_cfg)])
    fresh_norm = VecNormalize(fresh_raw, norm_obs=True, norm_reward=True)
    agent_vn2 = PPOAgent(
        fresh_norm.observation_space,
        fresh_norm.action_space,
        agent_cfg,
    )
    agent_vn2.load(_SAVE_PATH_VN, env=fresh_norm)
    assert agent_vn2.model is not None
    assert isinstance(agent_vn2.env, VecNormalize)
    assert not agent_vn2.env.training, "VecNormalize should be frozen after load"
    ok()
except Exception as e:
    fail(e)

# ── 8. evaluate() ────────────────────────────────────────
print("\n[8] evaluate() — 2 deterministic episodes")
try:
    step("Build single (non-vectorized) eval env")
    eval_env = AuctionNetGymEnv(env_cfg)
    ok()

    step("agent.evaluate(eval_env, n_eval_episodes=2)")
    metrics = agent.evaluate(eval_env, n_eval_episodes=2)
    assert set(metrics.keys()) == {"roi", "budget_utilization", "win_rate"}, \
        f"unexpected keys: {metrics.keys()}"
    assert all(isinstance(v, float) for v in metrics.values()), \
        "all metric values should be floats"
    assert metrics["budget_utilization"] >= 0.0
    assert metrics["win_rate"] >= 0.0
    ok()
    print(f"      roi                = {metrics['roi']:.4f}")
    print(f"      budget_utilization = {metrics['budget_utilization']:.4f}")
    print(f"      win_rate           = {metrics['win_rate']:.4f}")
except Exception as e:
    fail(e)

# ── Cleanup ──────────────────────────────────────────────
shutil.rmtree("saved_models", ignore_errors=True)

print("\n" + "=" * 55)
print("  All checks passed.")
print("=" * 55)
