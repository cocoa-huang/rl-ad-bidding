
**Adjust config files as needed!**

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Fixed Baseline Evaluation

```bash
python -m scripts.evaluate --config configs/fixed.yaml --agent-type fixed --episodes 5
```

---

## 3. PPO Local training (smoke test)

```bash
python -m scripts.train --config configs/default.yaml --agent-type ppo --run-name smoke_ppo --total-timesteps 500
```

---

## 4. PPO Evaluation

```bash
python -m scripts.evaluate --config configs/default.yaml --agent-type ppo --checkpoint saved_models/ppo_smoke_ppo_final --episodes 5
```

---

## 5. Collect offline dataset

```bash
python -m scripts.collect_offline_data
```

---

## 6. IQL Training (via local dataset)

```bash
python -m scripts.train --config configs/iql.yaml --agent-type iql --offline-dataset data/offline_dataset_debug.npz --run-name iql_debug
```

---

## 7. IQL Evaluation

```bash
python -m scripts.evaluate --config configs/iql.yaml --agent-type iql --checkpoint saved_models/iql_iql-debug_final.pt --episodes 5
```

---

## 8. Python 编译检查（快速排错）

```bash
python -m py_compile scripts/train.py scripts/evaluate.py agents/ppo_agent.py agents/fixed_bid_baseline.py agents/iql_agent.py
```

---