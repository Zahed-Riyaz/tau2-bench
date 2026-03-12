"""
RL fine-tuning experiment for airline_multistep using Tinker API.

Targets the chain-following gaps observed in the Llama-3.3-70B baseline (2/15):
  - Agent picks wrong reservation_id at step 2   (ms_a, ms_e failures)
  - Agent skips the search_direct_flight step     (ms_d failures)
  - Agent generates malformed tool-call JSON      (ms_e crash)
  - User simulator terminates immediately         (ms_b failures)

Pipeline
--------
  Phase 1  Collect rollouts on the TRAIN split via tau2 run (Groq for speed).
  Phase 2  Load saved trajectories, compute REINFORCE advantages, LoRA-tune
           the agent on Tinker using a weighted cross-entropy loss.
  Phase 3  Evaluate the tuned model on the TEST split and compare rewards.

Quick start
-----------
    export TINKER_API_KEY=<key>
    export GROQ_API_KEY=<key>
    python -m tau2.scripts.rl_airline_experiment

Skip rollout collection if you already have a simulation file:
    python -m tau2.scripts.rl_airline_experiment \\
        --skip-rollouts \\
        --trajectories-file data/tau2/simulations/<run>.json

Usage after LoRA training (post-RL eval only):
    python -m tau2.scripts.rl_airline_experiment \\
        --skip-rollouts --skip-train \\
        --tuned-model tinker://<run-id>/sampler_weights/final
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import tinker
    from tinker import AdamParams, Datum, EncodedTextChunk, ModelInput, SamplingParams

    TINKER_OK = True
except ImportError:
    TINKER_OK = False
    logger.warning("tinker not installed — training phases will be skipped. pip install tinker")

# ── Paths & constants ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _REPO_ROOT / "data" / "tau2"
SIM_DIR = DATA_DIR / "simulations"
SIM_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = "airline"
TASK_SET = "airline_multistep"

# Smallest capable instruction model available on Tinker with tool-calling support.
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Rollout collection model.
# Default: Groq (fast, free).  Pass --use-tinker-inference to use Tinker instead
# (no rate limits, same session as the training client).
ROLLOUT_MODEL = "groq/llama-3.3-70b-versatile"
USER_MODEL = "groq/llama-3.3-70b-versatile"

TINKER_OAI_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# RL hyper-params
LORA_RANK = 16
ADAM_LR = 3e-4
NUM_EPOCHS = 2
REWARD_BASELINE = 0.5   # shift {0,1} reward → {-0.5, +0.5} advantage


# ── Tinker inference setup ─────────────────────────────────────────────────────

def create_tinker_inference_checkpoint(training_client) -> tuple[str, dict]:
    """
    Save the current LoRA weights as a lightweight sampler checkpoint and return:
      (agent_llm_string, env_overrides)

    The agent_llm_string is passed to tau2 via --agent-llm.
    env_overrides patches OPENAI_API_BASE / OPENAI_API_KEY so litellm routes
    the call to Tinker's OpenAI-compatible endpoint instead of OpenAI.

    No rate limits: Tinker's inference API is billed by compute, not by RPM/TPM.
    """
    logger.info("Saving initial LoRA weights for Tinker inference …")
    checkpoint_path: str = training_client.save_weights_for_sampler("rollout_init").result().path
    logger.info(f"Tinker inference checkpoint → {checkpoint_path}")

    agent_llm = f"openai/{checkpoint_path}"
    env_overrides = {
        "OPENAI_API_BASE": TINKER_OAI_BASE,
        "OPENAI_API_KEY": os.environ.get("TINKER_API_KEY", ""),
    }
    return agent_llm, env_overrides


# ── Phase 1: Rollout collection ────────────────────────────────────────────────

def collect_rollouts(
    out_file: Path,
    agent_llm: str = ROLLOUT_MODEL,
    env_overrides: dict | None = None,
) -> Path:
    """
    Run tau2 on the TRAIN split and save trajectories to *out_file*.

    agent_llm      — the litellm model string for the agent.
    env_overrides  — extra environment variables (e.g. to point litellm at
                     Tinker's OpenAI endpoint instead of Groq).
    """
    logger.info(f"Phase 1 — collecting rollouts with agent={agent_llm} …")
    cmd = [
        sys.executable, "-m", "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", TASK_SET,
        "--task-split-name", "train",
        "--agent-llm", agent_llm,
        "--user-llm", USER_MODEL,
        "--save-to", str(out_file),
    ]
    env = {**os.environ, **(env_overrides or {})}
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.warning("tau2 run exited with non-zero status — some tasks may have failed.")
    logger.info(f"Trajectories saved → {out_file}")
    return out_file


# ── Phase 2: RL fine-tuning ────────────────────────────────────────────────────

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """
    Convert tau2 Message dicts (already serialised) to OpenAI-style dicts
    that HuggingFace tokenizers accept for apply_chat_template.
    """
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": m.get("content") or ""})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": m.get("content") or "",
            }
            if m.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            out.append(entry)
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("id", ""),
                "content": m.get("content") or "",
            })
    return out


def _encode_turn(
    tokenizer,
    context: list[dict],
    response: dict,
) -> tuple[list[int], list[int]]:
    """
    Tokenise a single assistant turn.

    Returns:
        context_tokens  — the prompt tokens (will get weight=0)
        response_tokens — the completion tokens (will get weight=advantage)
    """
    ctx_str = tokenizer.apply_chat_template(
        context,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_str = tokenizer.apply_chat_template(
        context + [response],
        tokenize=False,
        add_generation_prompt=False,
    )
    ctx_tokens: list[int] = tokenizer.encode(ctx_str, add_special_tokens=False)
    full_tokens: list[int] = tokenizer.encode(full_str, add_special_tokens=False)
    resp_tokens = full_tokens[len(ctx_tokens):]
    return ctx_tokens, resp_tokens


def build_datums(
    simulation_results: list[dict],
    tokenizer,
    baseline: float = REWARD_BASELINE,
) -> list[Datum]:
    """
    Convert tau2 SimulationRun dicts → Tinker Datum objects.

    Strategy (REINFORCE with constant baseline):
      advantage = reward - baseline
      Each assistant turn in the episode gets the same advantage as a
      per-token weight.  Positive advantage → reinforce those tokens.
      Negative advantage → suppress them.
    """
    datums: list[Datum] = []

    for sim in simulation_results:
        reward: float = sim.get("reward", 0.0)
        advantage = reward - baseline
        messages_raw: list[dict] = sim.get("messages", [])

        if not messages_raw:
            continue

        # Skip episodes that are trivially uninformative (all tokens 0 weight)
        if abs(advantage) < 1e-6:
            logger.debug(f"Task {sim.get('task_id')}: advantage≈0, skipping.")
            continue

        oai_messages = _to_openai_messages(messages_raw)

        for i, msg in enumerate(oai_messages):
            if msg["role"] != "assistant":
                continue

            context = oai_messages[:i]
            if not context:
                continue

            try:
                ctx_tokens, resp_tokens = _encode_turn(tokenizer, context, msg)
            except Exception as e:
                logger.warning(f"Tokenisation failed for task {sim.get('task_id')} turn {i}: {e}")
                continue

            if not resp_tokens:
                continue

            all_tokens = ctx_tokens + resp_tokens
            # weights: 0 for prompt, advantage for completion
            weights = [0.0] * len(ctx_tokens) + [advantage] * len(resp_tokens)

            datum = Datum(
                model_input=ModelInput(
                    chunks=[EncodedTextChunk(tokens=all_tokens)]
                ),
                loss_fn_inputs={
                    "targets": all_tokens,
                    "weights": weights,
                },
            )
            datums.append(datum)
            logger.debug(
                f"Task {sim.get('task_id')} turn {i}: "
                f"reward={reward:.2f} advantage={advantage:+.2f} "
                f"resp_len={len(resp_tokens)}"
            )

    logger.info(f"Built {len(datums)} training datums from {len(simulation_results)} trajectories.")
    return datums


def create_training_client():
    """Create (and return) a Tinker LoRA training client for BASE_MODEL."""
    if not TINKER_OK:
        raise ImportError("tinker package not installed. Run: pip install tinker")
    svc = tinker.ServiceClient()
    client = svc.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
    ).result()
    logger.info(f"LoRA training session started — base model: {BASE_MODEL}, rank={LORA_RANK}")
    return client


def run_rl_training(trajectories_file: Path, training_client=None) -> str:
    """
    Fine-tune BASE_MODEL on the collected trajectories.

    training_client — pass an existing client to reuse the same Tinker session
                      (e.g. when --use-tinker-inference was used for rollouts).
                      If None, a fresh session is created.

    Returns the tinker:// path of the saved sampler checkpoint.
    """
    if not TINKER_OK:
        raise ImportError("tinker package not installed. Run: pip install tinker")

    logger.info("Phase 2 — RL fine-tuning via Tinker …")

    # Load trajectories
    with open(trajectories_file) as f:
        data = json.load(f)
    simulations: list[dict] = data.get("simulations", [])
    logger.info(f"Loaded {len(simulations)} simulation runs from {trajectories_file}")

    # Reuse or create a training client
    if training_client is None:
        training_client = create_training_client()

    # Get tokenizer from the training client
    tokenizer = training_client.get_tokenizer().result()

    # Build Datum objects
    datums = build_datums(simulations, tokenizer)
    if not datums:
        raise ValueError(
            "No training datums were produced. "
            "Check that trajectories_file contains valid simulation runs."
        )

    # RL training loop
    adam = AdamParams(
        lr=ADAM_LR,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} — {len(datums)} datums …")
        total_loss = 0.0

        for step, datum in enumerate(datums):
            # Tinker operates on ~10-second clock cycles.
            # Submit forward_backward, then immediately submit optim_step
            # to overlap them within the same cycle.
            fwd = training_client.forward_backward(datum, loss_fn="cross_entropy")
            opt = training_client.optim_step(adam)

            fwd_result = fwd.result()
            opt.result()

            loss = fwd_result.loss if hasattr(fwd_result, "loss") else float("nan")
            total_loss += loss if loss == loss else 0.0  # skip NaN

            if (step + 1) % 5 == 0:
                logger.info(
                    f"  Epoch {epoch + 1} step {step + 1}/{len(datums)} "
                    f"avg_loss={total_loss / (step + 1):.4f}"
                )

        logger.info(f"Epoch {epoch + 1} done — avg_loss={total_loss / len(datums):.4f}")

    # Save checkpoint
    checkpoint_name = f"airline_rl_e{NUM_EPOCHS}"
    save_result = training_client.save_weights_for_sampler(checkpoint_name).result()
    tuned_path: str = save_result.path
    logger.info(f"Checkpoint saved → {tuned_path}")
    return tuned_path


# ── Phase 3: Post-RL evaluation ────────────────────────────────────────────────

def run_post_eval(tuned_model_path: str, out_file: Path) -> None:
    """
    Evaluate the tuned model on the TEST split via tau2.

    Uses Tinker's OpenAI-compatible endpoint so no code changes to tau2 are needed.
    """
    logger.info("Phase 3 — post-RL evaluation on test split …")

    tinker_base_url = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    tinker_api_key = os.environ.get("TINKER_API_KEY", "")

    # litellm supports custom OpenAI-compatible providers via the openai/ prefix
    # and OPENAI_API_BASE / OPENAI_API_KEY env vars.
    env = {
        **os.environ,
        "OPENAI_API_BASE": tinker_base_url,
        "OPENAI_API_KEY": tinker_api_key,
    }
    agent_llm = f"openai/{tuned_model_path}"

    cmd = [
        sys.executable, "-m", "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", TASK_SET,
        "--task-split-name", "test",
        "--agent-llm", agent_llm,
        "--user-llm", USER_MODEL,   # keep user sim on Groq
        "--save-to", str(out_file),
    ]
    logger.info(f"Running post-RL eval: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)
    logger.info(f"Post-RL results saved → {out_file}")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_comparison(before_file: Path | None, after_file: Path | None) -> None:
    """Print a simple reward comparison table."""

    def _load(path: Path | None) -> dict[str, float]:
        if path is None or not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        return {
            sim["task_id"]: sim.get("reward", float("nan"))
            for sim in data.get("simulations", [])
        }

    before = _load(before_file)
    after = _load(after_file)
    all_ids = sorted(set(before) | set(after))

    print("\n" + "=" * 55)
    print(f"{'Task':<12}  {'Before (baseline)':>18}  {'After (RL)':>12}")
    print("-" * 55)
    for tid in all_ids:
        b = before.get(tid, float("nan"))
        a = after.get(tid, float("nan"))
        arrow = "→"
        if a > b:
            arrow = "↑"
        elif a < b:
            arrow = "↓"
        print(f"{tid:<12}  {b:>18.4f}  {arrow} {a:>10.4f}")
    print("=" * 55)

    if before and after:
        avg_b = sum(before.values()) / len(before)
        avg_a = sum(after.values()) / len(after)
        print(f"{'AVERAGE':<12}  {avg_b:>18.4f}  → {avg_a:>10.4f}")
    print()


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL fine-tuning experiment for airline_multistep via Tinker."
    )
    parser.add_argument(
        "--skip-rollouts", action="store_true",
        help="Skip Phase 1 (rollout collection). Requires --trajectories-file.",
    )
    parser.add_argument(
        "--trajectories-file", type=Path, default=None,
        help="Path to a previously saved tau2 simulation JSON (used when --skip-rollouts).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip Phase 2 (RL training). Requires --tuned-model.",
    )
    parser.add_argument(
        "--tuned-model", type=str, default=None,
        help="tinker:// path to a pre-trained checkpoint (used when --skip-train).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip Phase 3 (post-RL evaluation).",
    )
    parser.add_argument(
        "--use-tinker-inference", action="store_true",
        help=(
            "Use Tinker's OpenAI-compatible endpoint for rollout collection "
            "instead of Groq. Eliminates RPM/TPM rate limits. "
            "Requires TINKER_API_KEY."
        ),
    )
    args = parser.parse_args()

    ts = int(time.time())
    rollout_file = args.trajectories_file or (SIM_DIR / f"airline_ms_train_rollouts_{ts}.json")
    post_eval_file = SIM_DIR / f"airline_ms_test_post_rl_{ts}.json"

    # Shared training client (created once when --use-tinker-inference so that
    # rollout collection and RL training reuse the same Tinker session / weights).
    shared_training_client = None

    # Phase 1
    if not args.skip_rollouts:
        rollout_agent_llm = ROLLOUT_MODEL
        rollout_env: dict | None = None

        if args.use_tinker_inference:
            if not TINKER_OK:
                parser.error("--use-tinker-inference requires the tinker package. pip install tinker")
            shared_training_client = create_training_client()
            rollout_agent_llm, rollout_env = create_tinker_inference_checkpoint(shared_training_client)
            logger.info("Using Tinker inference for rollouts — no rate limits.")

        collect_rollouts(rollout_file, agent_llm=rollout_agent_llm, env_overrides=rollout_env)
    elif not rollout_file.exists():
        parser.error(f"--trajectories-file {rollout_file} does not exist.")

    # Phase 2
    tuned_model = args.tuned_model
    if not args.skip_train:
        tuned_model = run_rl_training(rollout_file, training_client=shared_training_client)
        logger.info(f"Tuned model checkpoint: {tuned_model}")
    elif tuned_model is None:
        parser.error("--skip-train requires --tuned-model <tinker://…>.")

    # Phase 3
    if not args.skip_eval:
        run_post_eval(tuned_model, post_eval_file)

    # Summary
    print_comparison(rollout_file, post_eval_file if not args.skip_eval else None)
    logger.info("Done.")


if __name__ == "__main__":
    main()
