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
GROQ_API_KEY=gsk_...          # for Groq models
ANTHROPIC_API_KEY=sk-ant-...  # Claude models
OPENAI_API_KEY=sk-...         # GPT models
HF_TOKEN=hf_...               # HuggingFace Hub (only needed for --push-to-hub)
```

The `.env` file is loaded automatically when tau2 starts.

### Which key do I need?

| Goal | Key needed |
|------|-----------|
| Run benchmark with Groq | `GROQ_API_KEY` |
| Run benchmark with Claude | `ANTHROPIC_API_KEY` |
| Run benchmark with GPT | `OPENAI_API_KEY` |
| RL fine-tuning + Phase 3 eval | any of the above + `HF_TOKEN` |

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
# Groq
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm groq/llama-3.3-70b-versatile --user-llm groq/llama-3.3-70b-versatile

# Claude
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm claude-sonnet-4-6 --user-llm claude-sonnet-4-6

# OpenAI
tau2 run --domain airline --task-set-name airline_multistep --task-ids ms_a_0 --agent-llm gpt-4.1 --user-llm gpt-4.1
```

Task ID reference:

| ID | Type | Tool call sequence |
|----|------|--------------------|
| `ms_a_0` `ms_a_1` `ms_a_2` | A — Flight Status | `get_user_details` → `get_reservation_details` → `get_flight_status` |
| `ms_b_0` `ms_b_1` `ms_b_2` | B — Compensation  | `get_user_details` → `get_reservation_details` → `get_flight_status` → `send_certificate` |
| `ms_c_0` `ms_c_1` `ms_c_2` | C — Search→Book   | `search_direct_flight` → `get_user_details` → `book_reservation` |
| `ms_d_0` `ms_d_1` `ms_d_2` | D — Upgrade       | `get_reservation_details` → `search_direct_flight` → `update_reservation_flights` |
| `ms_e_0` `ms_e_1` `ms_e_2` | E — Add Baggage   | `get_user_details` → `get_reservation_details` → `update_reservation_baggages` |

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
# Groq
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
export GROQ_API_KEY=<your-groq-key>        # rollout collection + user simulator
export HF_TOKEN=<your-hf-token>            # required for --push-to-hub

# Optional — only needed if using Claude or GPT as the rollout agent
export ANTHROPIC_API_KEY=<your-claude-key> # --rollout-model claude-sonnet-4-6
export OPENAI_API_KEY=<your-openai-key>    # --rollout-model gpt-4.1
```

### Step 3 — Run the full pipeline

```bash
# Groq
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

### Step 5 — Disable 4-bit quantisation (required on Mac / CPU)

4-bit quantisation requires a CUDA GPU with bitsandbytes. On Mac or any CPU-only
machine, use `--no-quantize` which loads the model in **float32**:

```bash
python -m tau2.scripts.rl_airline_experiment --no-quantize --push-to-hub <repo>
```

> **Why float32?** float16 and bfloat16 produce NaN gradients during the backward
> pass on CPU. The script includes NaN guards that skip corrupted steps, but
> float32 is the only dtype that trains stably without a GPU.

### Step 6 — Reuse an existing baseline file (skip Phase 0)

If you have already run a test-split baseline and want to skip re-running it:

```bash
python -m tau2.scripts.rl_airline_experiment \
    --skip-rollouts \
    --trajectories-file data/simulations/airline_ms_train_<ts>.json \
    --baseline-file data/simulations/airline_ms_baseline_test_<ts>.json \
    --push-to-hub <your-hf-username>/airline-rl-tuned
```

### All flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--skip-rollouts` | off | Skip Phases 0 & 1; requires `--trajectories-file` |
| `--trajectories-file <path>` | auto-generated | Path to a saved tau2 simulation JSON (train split) |
| `--baseline-file <path>` | auto-generated | Path to an existing test-split baseline JSON; skips Phase 0 |
| `--model-output-dir <path>` | `output/airline_rl_tuned` | Where to save the tuned model |
| `--push-to-hub <repo-id>` | off | Push tuned model to HF Hub and run Phase 3 eval |
| `--no-quantize` | off | Use float32 instead of 4-bit quantisation (required on CPU/Mac) |
| `--rollout-model <model>` | `groq/llama-3.3-70b-versatile` | LiteLLM model for rollout collection and baseline |
| `--user-model <model>` | `groq/llama-3.3-70b-versatile` | LiteLLM model for the user simulator |

### Key hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | Swap to `Qwen2.5-7B-Instruct` for real results |
| LoRA rank | 16 | ~0.4% of parameters are trainable |
| Learning rate | 1e-5 | Kept low for stable convergence; 3e-4 causes NaN on first step |
| Reward baseline | 0.5 | Shifts `{0,1}` rewards to `{-0.5, +0.5}` advantages |
| Epochs | 2 | |
| Max sequence length | 512 tokens | Sequences truncated from the left (keeps recent context) |

### Phase 3 limitation — HuggingFace Inference API

The free HuggingFace Inference API only serves a curated set of popular models.
A custom fine-tuned model pushed to Hub will return:

```
model_not_supported: The requested model is not supported by any provider you have enabled
```

To run Phase 3 you need one of:
- **HF Inference Endpoints** (paid, dedicated) — deploy your model at
  huggingface.co/inference-endpoints, then pass `--agent-llm-args` with the endpoint URL
- **Together AI / Replicate** — deploy the model and point tau2 at the OpenAI-compatible API
- A Colab T4 GPU running vllm to serve the model locally

---

