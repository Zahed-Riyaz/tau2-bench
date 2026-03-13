# Airline Multi-Step Tasks — Run Guide

This guide covers how to run, play, evaluate, and RL-tune against the
`airline_multistep` task set introduced on the `airline-tasks` branch.

---

## 1. Installation

### Step 1 — Clone and install the package

```bash
git clone <repo-url>
cd tau2-bench
conda create -n tau2 python=3.11 -y
conda activate tau2
pip install -e .
```

### Step 2 — Install optional dependencies

```bash
# Tinker (RL fine-tuning)
pip install tinker

# Verify the data files are present
tau2 check-data
```

---

## 2. API Keys

Create a `.env` file at the repo root and add whichever keys you have:

```bash
# .env
GROQ_API_KEY=gsk_...          # free tier, recommended for testing
ANTHROPIC_API_KEY=sk-ant-...  # Claude models
OPENAI_API_KEY=sk-...         # GPT models
TINKER_API_KEY=...            # RL fine-tuning via Tinker
```

The `.env` file is loaded automatically when tau2 starts.

### Which key do I need?

| Goal | Key needed |
|------|-----------|
| Run benchmark (recommended) | `GROQ_API_KEY` |
| Use Claude as agent/user | `ANTHROPIC_API_KEY` |
| Use GPT as agent/user | `OPENAI_API_KEY` |
| RL fine-tuning | `TINKER_API_KEY` |

---

## 3. Regenerate Task Files (optional)

The generated files are already committed. Re-run only if you change the generator.

```bash
python -m tau2.domains.airline.tasks.create_multistep_tasks
```

Expected output:

```
Type A (Flight Status Discovery):   3 tasks
Type B (Cancellation Compensation): 3 tasks
Type C (Search-then-Book):          3 tasks
Type D (Flight Update):             3 tasks
Type E (Add Baggage):               3 tasks
Total:                              15 tasks
```

---

## 4. Interactive Conversation (`tau2 play`)

The quickest way to test a single conversation manually.

```bash
tau2 play
```

At the prompts:
1. **Domain** → select `airline`
2. **Task set** → select `airline_multistep`
3. **Task** → pick any `ms_a_*` through `ms_e_*` task
4. **Role** → `agent` (you type the agent's responses) or `user` (you type as the user, an LLM plays the agent)

---

## 5. Run a Single Task (automated)

```bash
# Groq (recommended — fast, free)
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile

# Claude
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm claude-sonnet-4-6 --user-llm claude-sonnet-4-6

# OpenAI
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm gpt-4.1 --user-llm gpt-4.1
```

Task ID reference:

| ID | Type | Dependency chain |
|----|------|-----------------|
| `ms_a_0` `ms_a_1` `ms_a_2` | A — Flight Status | user_id → reservation → flight_number → status |
| `ms_b_0` `ms_b_1` `ms_b_2` | B — Compensation  | user_id → reservation → disrupted_flight → send_certificate |
| `ms_c_0` `ms_c_1` `ms_c_2` | C — Search→Book   | search_flight → user_details → book_reservation |
| `ms_d_0` `ms_d_1` `ms_d_2` | D — Upgrade       | reservation_details → search_flight → update_flights |
| `ms_e_0` `ms_e_1` `ms_e_2` | E — Add Baggage   | user_details → reservation → update_baggages |

---

## 6. Run the Full Task Set (all 15 tasks)

```bash
# All 15 tasks — base split
tau2 run --domain airline --task-set-name airline_multistep --task-split-name base --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile

# Train split only (10 tasks)
tau2 run --domain airline --task-set-name airline_multistep --task-split-name train --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile

# Test split only (5 tasks)
tau2 run --domain airline --task-set-name airline_multistep --task-split-name test --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile
```

Save results to a file for later analysis:

```bash
tau2 run --domain airline --task-set-name airline_multistep --task-split-name base --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile --save-to data/tau2/simulations/baseline.json
```

---

## 7. Model Options

LiteLLM is used under the hood, so any model string it supports works:

```bash
# Groq — free, fast, no download (recommended)
--agent-llm groq/llama-3.3-70b-versatile

# Anthropic Claude
--agent-llm claude-sonnet-4-6
--agent-llm claude-opus-4-6

# OpenAI
--agent-llm gpt-4.1
--agent-llm gpt-4.1-mini

# Mixed: strong agent, cheaper user simulator
--agent-llm claude-sonnet-4-6 --user-llm groq/llama-3.3-70b-versatile
```

---

## 8. View Results

```bash
# Interactive viewer for all saved runs
tau2 view

# Point at a specific file
tau2 view --file data/tau2/simulations/baseline.json

# Show only failed tasks
tau2 view --only-show-failed
```

---

## 9. Re-evaluate Rewards on Saved Trajectories

Useful after tweaking evaluation criteria without re-running conversations:

```bash
tau2 evaluate-trajs data/tau2/simulations/baseline.json
```

---

## 10. Run the Test Suite

Validates task structure, DB references, dependency chains, splits, and registry:

```bash
pytest tests/test_airline_multistep.py -v
```

All 34 tests should pass.

---

## 11. RL Fine-Tuning (HuggingFace PEFT — free, runs on Colab T4)

The RL script targets the chain-following gaps in weaker models (wrong reservation IDs,
skipped search steps, malformed tool calls). It runs three phases automatically:

| Phase | What happens |
|-------|-------------|
| 1 — Rollout collection | Runs tau2 on the train split via Groq, saves trajectories |
| 2 — RL training | REINFORCE with advantage-weighted cross-entropy, LoRA via HuggingFace PEFT |
| 3 — Post-RL eval | Evaluates tuned model on test split, prints before/after table |

No paid API required — training runs locally or on a free Google Colab T4 GPU.
Base model: `Qwen/Qwen2.5-7B-Instruct` (open, no HuggingFace access gate required). To use Llama instead, run `huggingface-cli login` and set `BASE_MODEL` in the script.

### Step 1 — Install dependencies

```bash
pip install transformers peft accelerate torch bitsandbytes
```

### Step 2 — Set API keys

```bash
export GROQ_API_KEY=<your-groq-key>      # rollout collection + user simulator
# export ANTHROPIC_API_KEY=<key>         # optional: use Claude as user sim
# export OPENAI_API_KEY=<key>            # optional: use GPT as user sim
# export HF_TOKEN=<key>                  # only needed for --push-to-hub
```

### Step 3 — Run the full pipeline

```bash
python -m tau2.scripts.rl_airline_experiment
```

Saves the tuned model to `output/airline_rl_tuned/` by default.

### Step 4 — Skip rollout collection (reuse existing simulation file)

If you already ran `tau2 run --save-to ...`, pass that file directly:

```bash
python -m tau2.scripts.rl_airline_experiment \
    --skip-rollouts \
    --trajectories-file data/tau2/simulations/baseline.json
```

### Step 5 — Skip training, only run post-RL evaluation

```bash
python -m tau2.scripts.rl_airline_experiment \
    --skip-rollouts \
    --skip-train \
    --model-output-dir output/airline_rl_tuned \
    --trajectories-file data/tau2/simulations/baseline.json
```

### Step 6 — Push tuned model to HuggingFace Hub

Useful for sharing or loading the model on another machine:

```bash
python -m tau2.scripts.rl_airline_experiment \
    --push-to-hub <your-hf-username>/airline-rl-tuned
```

After pushing, post-RL eval automatically uses the Hub model ID.

### Step 7 — Disable 4-bit quantisation (if bitsandbytes unavailable)

```bash
python -m tau2.scripts.rl_airline_experiment --no-quantize
```

### Step 8 — Skip evaluation (train only)

```bash
python -m tau2.scripts.rl_airline_experiment --skip-eval
```

### Step 9 — Filter out failed trajectories (reward-weighted regression)

Only train on successful episodes — useful when failures dominate the dataset and negative gradient is noisy:

```bash
python -m tau2.scripts.rl_airline_experiment --filter-failed
```

### Step 10 — Run multiple on-policy iterations

After the first iteration the trained model generates its own rollouts, reducing the off-policy mismatch between the 70B rollout model and the 8B training model:

```bash
python -m tau2.scripts.rl_airline_experiment --rl-iterations 3
```

Each iteration saves a separate rollout file and model checkpoint (`…_iter0`, `…_iter1`, …).

### All flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-rollouts` | off | Skip Phase 1; requires `--trajectories-file` |
| `--trajectories-file <path>` | auto-generated | Path to a saved tau2 simulation JSON |
| `--skip-train` | off | Skip Phase 2; requires `--model-output-dir` to exist |
| `--model-output-dir <path>` | `output/airline_rl_tuned` | Where to save/load the tuned model |
| `--skip-eval` | off | Skip Phase 3 post-RL evaluation |
| `--push-to-hub <repo-id>` | off | Push tuned model to HuggingFace Hub after training |
| `--no-quantize` | off | Use float16 instead of 4-bit (needs more VRAM) |
| `--filter-failed` | off | Skip reward=0 trajectories; trains only on successes |
| `--rl-iterations <N>` | 1 | On-policy improvement loop; trained model generates rollouts from iteration 1 onward |
| `--no-normalise` | off | Disable group-normalised advantages; fall back to fixed baseline=0.5 |
| `--no-upweight` | off | Disable tool-call token upweighting (all response tokens weighted equally) |

### Output

The script prints a comparison table at the end:

```
=======================================================
Task          Before (baseline)         After (RL)
-------------------------------------------------------
ms_a_2               0.0000         →     0.0000
ms_b_0               0.0000         ↑     1.0000
ms_c_0               0.0000         ↑     1.0000
ms_d_0               0.0000         ↑     0.5000
ms_e_2               0.0000         ↑     1.0000
=======================================================
AVERAGE              0.0000         →     0.7000
```

---

## 12. Programmatic Usage

```python
from tau2.registry import registry
from tau2.run import run_task

# Load a single task
tasks = registry.get_tasks_loader("airline_multistep")("base")
task = next(t for t in tasks if t.id == "ms_b_0")

# Run it
sim = run_task(
    domain="airline",
    task=task,
    agent="llm_agent",
    user="user_simulator",
    llm_agent="groq/llama-3.3-70b-versatile",
    llm_user="groq/llama-3.3-70b-versatile",
    max_steps=200,
    seed=42,
)

print(sim.reward)
for msg in sim.messages:
    print(f"[{msg.role}] {msg.content}")
```

---

## Task Design — Why Intermediate IDs Matter

Every task tells the user simulator what it **does not know**
(`unknown_info` field). The user cannot volunteer a `reservation_id`
or `flight_number` it was never given. An agent that hallucinates any
intermediate ID will either get a tool error or produce an argument
mismatch against the ground-truth `actions` list — both resulting in
a failed evaluation.

The reward basis per type:

| Type | `reward_basis` | What is checked |
|------|---------------|-----------------|
| A    | ACTION + COMMUNICATE | Correct tool chain **and** status reported to user |
| B    | ACTION | All 4 tool calls with correct args (especially `send_certificate.amount`) |
| C    | ACTION | `book_reservation.flights` flight_number must match search output |
| D    | ACTION | `update_reservation_flights` must use route/date from reservation |
| E    | ACTION | Baggage update args correct |