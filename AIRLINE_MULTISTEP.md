# Airline Multi-Step Tasks — Run Guide

This guide covers how to run, play, and evaluate the `airline_multistep` task set
introduced on the `airline-tasks` branch.

---

## 1. Prerequisites

### Install the package

```bash
conda activate tau2
pip install -e .
```

### Set your API key

```bash
cp .env.example .env
# then edit .env and add your key:
# OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY=...
```

Verify everything is wired up:

```bash
tau2 check-data
```

---

## 2. Regenerate the Task Files (optional)

The generated files are already committed. Re-run only if you change the generator.

```bash
conda run -n tau2 python -m tau2.domains.airline.tasks.create_multistep_tasks
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

## 3. Interactive Conversation (`tau2 play`)

The quickest way to converse. The CLI will prompt you to choose a domain, role, and task.

```bash
conda run -n tau2 tau2 play
```

At the prompts:
1. **Domain** → select `airline`
2. **Task set** → select `airline_multistep`
3. **Task** → pick any `ms_a_*` through `ms_e_*` task
4. **Role** → `agent` (you type the agent's responses) or `user` (you type as the user, an LLM plays the agent)

When playing as **user** against an LLM agent, set your agent model:

```bash
# The play command will ask interactively; alternatively, use the gym interface below.
```

---

## 4. Run a Single Task (automated, one conversation)

Run one specific task end-to-end with two LLMs:

```bash
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-ids ms_b_0 \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-trials 1
```

Swap any task ID from the list below:

| ID | Type | Chain |
|----|------|-------|
| `ms_a_0` `ms_a_1` `ms_a_2` | A — Flight Status | user_id → reservation → flight_number → status |
| `ms_b_0` `ms_b_1` `ms_b_2` | B — Compensation  | user_id → reservation → disrupted_flight → send_certificate |
| `ms_c_0` `ms_c_1` `ms_c_2` | C — Search→Book   | search_flight → user_details → book_reservation |
| `ms_d_0` `ms_d_1` `ms_d_2` | D — Upgrade       | reservation_details → search_flight → update_flights |
| `ms_e_0` `ms_e_1` `ms_e_2` | E — Add Baggage   | user_details → reservation → update_baggages |

---

## 5. Run the Full Task Set (all 15 tasks)

```bash
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-split-name base \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --max-concurrency 3
```

### Train split only (9 tasks)

```bash
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-split-name train \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1
```

### Test split only (6 tasks)

```bash
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-split-name test \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1
```

Results are saved automatically to `data/tau2/simulations/`.

---

## 6. Use a Different Model

LiteLLM is used under the hood, so any model string it supports works:

```bash
# Claude
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-split-name base \
  --agent-llm claude-sonnet-4-6 \
  --user-llm claude-sonnet-4-6

# Mix agent and user models
conda run -n tau2 tau2 run \
  --task-set-name airline_multistep \
  --task-split-name base \
  --agent-llm claude-sonnet-4-6 \
  --user-llm gpt-4.1
```

---

## 7. View Results

```bash
# Interactive viewer for all saved runs
conda run -n tau2 tau2 view

# Point at a specific file
conda run -n tau2 tau2 view --file data/tau2/simulations/<filename>.json

# Show only failed tasks
conda run -n tau2 tau2 view --only-show-failed
```

---

## 8. Re-evaluate Rewards on Saved Trajectories

Useful after tweaking evaluation criteria without re-running conversations:

```bash
conda run -n tau2 tau2 evaluate-trajs data/tau2/simulations/<filename>.json
```

---

## 9. Run the Test Suite

Validates task structure, DB references, dependency chains, splits, and registry:

```bash
conda run -n tau2 pytest tests/test_airline_multistep.py -v
```

All 34 tests should pass.

---

## 10. Programmatic Usage

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
    llm_agent="gpt-4.1",
    llm_user="gpt-4.1",
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
| E    | ACTION + COMMUNICATE (if paid) | Baggage update args **and** cost communicated |
