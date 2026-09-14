# Reinforcement-Learning Workflow

1. Build/checksum with `make rom-test`.
2. Select a named state and record its hash in `state_manifests/`.
3. Define task success/failure events and a time limit.
4. Verify event traces before tuning rewards.
5. Run `make validate`, then train.

```bash
python -m pip install -e '.[train]'
python scripts/train.py --steps 100000
```

For each experiment record seed, commit, ROM MD5, state SHA-256, task/reward config, frame skip, and package versions. `artifacts/` and `runs/` are ignored. Separate train/evaluation states and treat RAM state as privileged information unless explicitly studying state-based policies.
