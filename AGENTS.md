# Agent Notes

## Project Mission

This checkout is not being used to prepare upstream changes for the original
LADX disassembly project. It uses the disassembly assets and symbols to build a
reproducible Zelda benchmark for reinforcement-learning and agent research.

The initial game is Link's Awakening DX. The main research directions are:

- skill transfer between rooms with related mechanics;
- extraction and evaluation of symbolic game logic from pixels, RAM symbols,
  state transitions, and disassembly source.

Benchmark quality, reproducibility, and clear experimental boundaries take
priority over adding many features.

## Design Constraints

- Keep code small, readable, and explicit. Prefer a short direct implementation
  over speculative abstractions, compatibility layers, or framework machinery.
- Add a dependency only when it provides clear benchmark value.
- Keep `ZeldaEnv` game-runtime focused and independent of individual tasks.
- Put reusable reward, success, and failure logic in `zelda_env/tasks/`.
- Put room/state splits and experiment parameters in `configs/experiments/`.
- Put reverse-engineering and memory-discovery tools in `zelda_env/utils/` or
  `scripts/`; they must not run implicitly during training.
- Do not recreate multi-game/backend abstractions until a second supported game
  or emulator creates a concrete requirement.

## Benchmark Contract

- Preserve the Gymnasium API: `reset()` returns `(observation, info)` and
  `step()` returns `(observation, reward, terminated, truncated, info)`.
- The default observation is the RGB pixel frame. RAM-derived state is
  privileged information exposed through `info["game_state"]` for rewards,
  diagnostics, and evaluation.
- Do not leak privileged state into a policy observation unless the experiment
  explicitly studies state-based or privileged policies and labels that result
  separately from the pixel benchmark.
- Keep the logical action mapping stable. Training may expose a documented
  subset, such as excluding `START` and `SELECT`.
- Avoid silently changing state fields, action IDs, task semantics, or metrics.
  Version intentional breaking changes.

## Memory and Symbolic Logic

- Resolve runtime addresses from `ladx-disassembly/azle.sym`; do not duplicate
  raw addresses throughout reward or task code.
- Keep the maintained semantic-field-to-symbol mapping centralized in
  `zelda_env/memory.py`.
- Treat `game_state["entities"]` as authoritative. Derived views such as
  `monsters` may be heuristic and must be documented as such.
- Confirm symbolic meanings with disassembly constants and code paths whenever
  possible. Record whether a fact is directly observed from RAM, statically
  derived from source, or only hypothesized from state changes.
- Prefer stable causal signals for task success, such as a counter or event flag
  change, over visual guesses or disappearance alone.

## Skill-Transfer Experiments

- Define skills by mechanics, not by room names. A room/save state is an
  instance of a reusable task.
- Keep training and evaluation rooms, states, reset variations, and seeds
  explicit in the experiment manifest.
- Prevent fixed-trajectory memorization by using held-out states or documented
  reset-time variation.
- Always retain a scratch baseline when measuring transfer or fine-tuning.
- Report task success rate and sample efficiency; also retain useful diagnostic
  metrics such as completion steps, damage, targets completed, and failure
  reasons.
- Use RAM only as privileged supervision unless the stated experiment is about
  symbolic/state observations.

## Reproducibility and Validation

- Record seeds, ROM checksum, save-state hashes, task configuration, and model
  initialization for benchmark runs.
- Do not edit a save state in place after it has been used in a recorded run.
- Reject byte-identical duplicate states; use `scripts/verify_states.py`.
- Before handing off a code change, run tests proportional to its scope. Changes
  to the environment or tasks should normally run `python -m pytest -q`, state
  verification, and a real-ROM smoke check.
- A training change should complete a short end-to-end smoke run without being
  represented as evidence that the task was learned.

## Generated and Reference Files

The repository and branch may contain local experiment files, emulator states,
screenshots, and other generated artifacts while this environment is being
developed.

- Keep unique save states that have a clear debugging or benchmark purpose.
- Keep generated ROM and symbol files available when needed for local tests, but
  do not treat them as source files.
- Do not commit caches, RAM dumps, screenshots, checkpoints, TensorBoard runs,
  or other generated outputs unless the user explicitly requests them.
- Preserve LADX reference material that supports room, entity, or symbolic-logic
  identification. Remove unrelated or superseded documentation.
- Use `make clean-cache` for routine cleanup; do not remove experiment results
  as part of a generic cache-clean command.

## GitHub Workflow

- Do not open pull requests against `haldai/LADX-Disassembly` or any upstream
  LADX disassembly repository unless the user explicitly asks for a PR.
- When asked to sync work to GitHub, commit and push only the requested files.
- Treat emulator state files, RAM dumps, screenshots, ROM outputs, and other
  generated artifacts as local unless the user explicitly asks to include them.
- Prefer preserving a dirty working tree over cleaning or reverting unrelated
  files.
