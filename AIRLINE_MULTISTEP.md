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

### Step 2 — Install optional dependencies (RL fine-tuning only)

```bash
pip install transformers peft accelerate torch bitsandbytes
```

---

## 2. API Keys

Create a `.env` file at the repo root and add whichever keys you have:

```bash
# .env
GROQ_API_KEY=gsk_...          # free tier, recommended for testing
ANTHROPIC_API_KEY=sk-ant-...  # Claude models
OPENAI_API_KEY=sk-...         # GPT models
HF_TOKEN=hf_...               # HuggingFace Hub (only needed for --push-to-hub)
```

The `.env` file is loaded automatically when tau2 starts.

### Which key do I need?

| Goal | Key needed |
|------|-----------|
| Run benchmark (recommended) | `GROQ_API_KEY` |
| Use Claude as agent/user | `ANTHROPIC_API_KEY` |
| Use GPT as agent/user | `OPENAI_API_KEY` |
| RL fine-tuning + Phase 3 eval | `GROQ_API_KEY` + `HF_TOKEN` |

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

> **Note**: if you installed via pip (e.g. on Colab) rather than `pip install -e .`,
> run this step after installation — the data files are not bundled with the package.

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

The RL script teaches a small open model to follow multi-step tool chains by
fine-tuning it on conversation trajectories collected from a stronger model (Groq).

| Phase | What happens |
|-------|-------------|
| 0 — Baseline | Scores the test split with Groq — the "Before" column |
| 1 — Rollouts | Runs tau2 on the train split via Groq, saves full conversation transcripts |
| 2 — RL training | REINFORCE with advantage-weighted cross-entropy, LoRA via HuggingFace PEFT |
| 3 — Post-RL eval | Pushes tuned model to HF Hub, scores test split, prints Before/After table |

Base model: `Qwen/Qwen2.5-0.5B-Instruct` (default, ~1 GB). Swap `BASE_MODEL` in
the script to `Qwen/Qwen2.5-7B-Instruct` for real results on a T4 GPU.

Phase 3 requires `--push-to-hub` — the tuned model is pushed to HuggingFace Hub
and then evaluated via tau2's standard LiteLLM routing. No vllm or local inference
server needed.

### Step 1 — Install dependencies

```bash
pip install transformers peft accelerate torch bitsandbytes
```

### Step 2 — Set API keys

```bash
export GROQ_API_KEY=<your-groq-key>        # rollout collection + user simulator (free)
export HF_TOKEN=<your-hf-token>            # required for --push-to-hub

# Optional — only needed if using Claude or GPT as the rollout agent
export ANTHROPIC_API_KEY=<your-claude-key> # --rollout-model claude-sonnet-4-6
export OPENAI_API_KEY=<your-openai-key>    # --rollout-model gpt-4.1
```

### Step 3 — Run the full pipeline

```bash
# Groq (free, recommended)
python -m tau2.scripts.rl_airline_experiment \
    --push-to-hub <your-hf-username>/airline-rl-tuned

# Anthropic Claude as rollout agent
python -m tau2.scripts.rl_airline_experiment \
    --rollout-model claude-sonnet-4-6 \
    --user-model groq/llama-3.3-70b-versatile \
    --push-to-hub <your-hf-username>/airline-rl-tuned

# OpenAI GPT as rollout agent
python -m tau2.scripts.rl_airline_experiment \
    --rollout-model gpt-4.1 \
    --user-model groq/llama-3.3-70b-versatile \
    --push-to-hub <your-hf-username>/airline-rl-tuned
```

This runs all four phases and prints the comparison table at the end.
Saves the tuned model to `output/airline_rl_tuned/` and pushes to HF Hub.

### Step 4 — Skip rollout collection (reuse existing simulation file)

If you already ran `tau2 run --save-to ...`, pass that file directly:

```bash
python -m tau2.scripts.rl_airline_experiment \
    --skip-rollouts \
    --trajectories-file data/tau2/simulations/baseline.json \
    --push-to-hub <your-hf-username>/airline-rl-tuned
```

### Step 5 — Disable 4-bit quantisation (if bitsandbytes unavailable)

```bash
python -m tau2.scripts.rl_airline_experiment --no-quantize --push-to-hub <repo>
```

### All flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-rollouts` | off | Skip Phases 0 & 1; requires `--trajectories-file` |
| `--trajectories-file <path>` | auto-generated | Path to a saved tau2 simulation JSON |
| `--model-output-dir <path>` | `output/airline_rl_tuned` | Where to save the tuned model |
| `--push-to-hub <repo-id>` | off | Push tuned model to HF Hub and run Phase 3 eval |
| `--no-quantize` | off | Use float16 instead of 4-bit (needs more VRAM) |
| `--rollout-model <model>` | `groq/llama-3.3-70b-versatile` | LiteLLM model for rollout collection and baseline |
| `--user-model <model>` | `groq/llama-3.3-70b-versatile` | LiteLLM model for the user simulator |

### Output

The script prints a comparison table at the end:

```
=========================================================
Task           Before          After
---------------------------------------------------------
ms_a_2         0.0000    →     0.0000
ms_b_2         0.0000    ↑     1.0000
ms_c_2         0.0000    ↑     1.0000
ms_d_2         0.0000    ↑     0.5000
ms_e_2         0.0000    ↑     1.0000
=========================================================
AVERAGE        0.0000    ↑     0.7000
```

Without `--push-to-hub`, the After column shows `N/A` and only the Before
baseline is printed.

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
