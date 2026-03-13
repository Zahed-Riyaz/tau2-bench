"""
RL fine-tuning experiment for airline_multistep using HuggingFace TRL + PEFT.

Targets the chain-following gaps observed in the Llama-3.3-70B baseline (2/15):
  - Agent picks wrong reservation_id at step 2   (ms_a, ms_e failures)
  - Agent skips the search_direct_flight step     (ms_d failures)
  - Agent generates malformed tool-call JSON      (ms_e crash)
  - User simulator terminates immediately         (ms_b failures)

Pipeline
--------
  Phase 1  Collect rollouts on the TRAIN split via tau2 run (Groq for speed).
  Phase 2  Load saved trajectories, compute REINFORCE advantages, LoRA-tune
           the agent locally using HuggingFace transformers + PEFT + PyTorch.
  Phase 3  Evaluate the tuned model on the TEST split and compare rewards.

Quick start (local GPU or Google Colab T4 — free)
-----------
    pip install transformers peft accelerate torch bitsandbytes
    export GROQ_API_KEY=<key>
    python -m tau2.scripts.rl_airline_experiment

Skip rollout collection if you already have a simulation file:
    python -m tau2.scripts.rl_airline_experiment \\
        --skip-rollouts \\
        --trajectories-file data/tau2/simulations/<run>.json

Post-RL eval only (tuned model already saved):
    python -m tau2.scripts.rl_airline_experiment \\
        --skip-rollouts --skip-train \\
        --model-output-dir ./output/tuned_model \\
        --trajectories-file data/tau2/simulations/<run>.json

Push tuned model to HuggingFace Hub for persistent storage:
    python -m tau2.scripts.rl_airline_experiment \\
        --push-to-hub <your-hf-username>/airline-rl-tuned
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

# ── Optional ML dependencies ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRL_OK = True
except ImportError:
    TRL_OK = False
    logger.warning(
        "transformers/peft/torch not installed — training phases will be skipped.\n"
        "Run: pip install transformers peft accelerate torch bitsandbytes"
    )

# ── Paths & constants ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _REPO_ROOT / "data" / "tau2"
SIM_DIR = DATA_DIR / "simulations"
SIM_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = "airline"
TASK_SET = "airline_multistep"

# Base model — small enough to fine-tune on a free Colab T4 (16 GB VRAM)
# with 4-bit quantisation + LoRA.
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Rollout collection — Groq is fast and free.
ROLLOUT_MODEL = "groq/llama-3.3-70b-versatile"
USER_MODEL = "groq/llama-3.3-70b-versatile"

# RL hyper-params
LORA_RANK = 16
LORA_ALPHA = 32
ADAM_LR = 3e-4
NUM_EPOCHS = 2
REWARD_BASELINE = 0.5   # shift {0,1} reward → {-0.5, +0.5} advantage
MAX_SEQ_LEN = 2048      # truncate long conversations to fit in VRAM


# ── Phase 1: Rollout collection ────────────────────────────────────────────────

def collect_rollouts(out_file: Path, agent_llm: str = ROLLOUT_MODEL) -> Path:
    """Run tau2 on the TRAIN split and save trajectories to *out_file*."""
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
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
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
        context_tokens  — the prompt tokens (will get weight=0 in training)
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


def build_training_tensors(
    simulation_results: list[dict],
    tokenizer,
    baseline: float = REWARD_BASELINE,
    max_len: int = MAX_SEQ_LEN,
) -> list[dict]:
    """
    Convert tau2 SimulationRun dicts → PyTorch training tensors.

    Strategy (REINFORCE with constant baseline):
      advantage = reward - baseline
      Each assistant turn in the episode gets the same advantage as a
      per-token weight. Positive advantage → reinforce those tokens.
      Negative advantage → suppress them.

    Returns a list of dicts with keys:
      input_ids  — LongTensor [seq_len]
      weights    — FloatTensor [seq_len]  (0 for prompt, ±advantage for response)
    """
    training_data = []

    for sim in simulation_results:
        reward: float = sim.get("reward", 0.0)
        advantage = reward - baseline
        messages_raw: list[dict] = sim.get("messages", [])

        if not messages_raw:
            continue

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
                logger.warning(
                    f"Tokenisation failed for task {sim.get('task_id')} turn {i}: {e}"
                )
                continue

            if not resp_tokens:
                continue

            all_tokens = ctx_tokens + resp_tokens
            weights = [0.0] * len(ctx_tokens) + [advantage] * len(resp_tokens)

            # Truncate to max_len from the right to keep recent context
            if len(all_tokens) > max_len:
                all_tokens = all_tokens[-max_len:]
                weights = weights[-max_len:]

            training_data.append({
                "input_ids": torch.tensor(all_tokens, dtype=torch.long),
                "weights": torch.tensor(weights, dtype=torch.float32),
            })
            logger.debug(
                f"Task {sim.get('task_id')} turn {i}: "
                f"reward={reward:.2f} advantage={advantage:+.2f} "
                f"seq_len={len(all_tokens)}"
            )

    logger.info(
        f"Built {len(training_data)} training tensors "
        f"from {len(simulation_results)} trajectories."
    )
    return training_data


def setup_model(quantize_4bit: bool = True):
    """
    Load BASE_MODEL with LoRA adapters.

    quantize_4bit=True uses bitsandbytes 4-bit quantisation — fits an 8B
    model inside a free Colab T4's 16 GB VRAM.  Set False on machines with
    more memory.
    """
    if not TRL_OK:
        raise ImportError(
            "transformers/peft/torch not installed.\n"
            "Run: pip install transformers peft accelerate torch bitsandbytes"
        )

    logger.info(f"Loading {BASE_MODEL} …")

    load_kwargs: dict[str, Any] = {"device_map": "auto"}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info(f"Model loaded with LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")
    return model, tokenizer


def run_rl_training(
    trajectories_file: Path,
    output_dir: Path,
    push_to_hub: str | None = None,
    quantize_4bit: bool = True,
) -> Path:
    """
    REINFORCE fine-tune BASE_MODEL on the collected trajectories.

    Uses:
      - HuggingFace transformers for model loading
      - PEFT LoRA for parameter-efficient fine-tuning
      - PyTorch manual training loop with advantage-weighted cross-entropy

    Saves the merged model to output_dir and optionally pushes to HF Hub.
    Returns output_dir.
    """
    logger.info("Phase 2 — RL fine-tuning via HuggingFace TRL + PEFT …")

    with open(trajectories_file) as f:
        data = json.load(f)
    simulations: list[dict] = data.get("simulations", [])
    logger.info(f"Loaded {len(simulations)} simulation runs.")

    model, tokenizer = setup_model(quantize_4bit=quantize_4bit)
    training_data = build_training_tensors(simulations, tokenizer)

    if not training_data:
        raise ValueError(
            "No training tensors produced. "
            "Check that trajectories_file contains valid simulation runs."
        )

    # Only fine-tune LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ADAM_LR,
        weight_decay=0.01,
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} — {len(training_data)} steps …")
        total_loss = 0.0

        for step, batch in enumerate(training_data):
            input_ids = batch["input_ids"].unsqueeze(0).to(model.device)   # [1, seq]
            weights = batch["weights"].unsqueeze(0).to(model.device)        # [1, seq]

            outputs = model(input_ids=input_ids)
            logits = outputs.logits                                          # [1, seq, vocab]

            # Causal LM: predict token[t+1] from token[t]
            shift_logits = logits[:, :-1, :].contiguous()                   # [1, seq-1, vocab]
            shift_labels = input_ids[:, 1:].contiguous()                    # [1, seq-1]
            shift_weights = weights[:, 1:].contiguous()                     # [1, seq-1]

            # Per-token cross-entropy
            token_losses = nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(1, -1)                                                    # [1, seq-1]

            # REINFORCE: weight losses by advantage
            # Normalise by sum of |weights| to keep loss scale stable
            denom = shift_weights.abs().sum() + 1e-8
            loss = (token_losses * shift_weights).sum() / denom

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 5 == 0:
                logger.info(
                    f"  Epoch {epoch + 1} step {step + 1}/{len(training_data)} "
                    f"avg_loss={total_loss / (step + 1):.4f}"
                )

        logger.info(
            f"Epoch {epoch + 1} done — avg_loss={total_loss / len(training_data):.4f}"
        )

    # Merge LoRA weights into base model and save
    logger.info(f"Merging LoRA weights and saving to {output_dir} …")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved → {output_dir}")

    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub → {push_to_hub} …")
        merged_model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        logger.info(f"Model available at: https://huggingface.co/{push_to_hub}")

    return output_dir


# ── Phase 3: Post-RL evaluation ────────────────────────────────────────────────

def run_post_eval(model_path: str, out_file: Path) -> None:
    """
    Evaluate the tuned model on the TEST split via tau2.

    model_path can be:
      - A local directory: ./output/tuned_model
      - A HuggingFace Hub model ID: username/model-name

    Uses litellm's huggingface/ provider so tau2 needs zero code changes.
    """
    logger.info(f"Phase 3 — post-RL evaluation on test split (model={model_path}) …")

    # litellm routes huggingface/<model> to the transformers pipeline locally
    # or to the HF Inference API if it's a Hub model ID.
    agent_llm = f"huggingface/{model_path}"

    cmd = [
        sys.executable, "-m", "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", TASK_SET,
        "--task-split-name", "test",
        "--agent-llm", agent_llm,
        "--user-llm", USER_MODEL,
        "--save-to", str(out_file),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
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
        description="REINFORCE RL fine-tuning for airline_multistep via HuggingFace PEFT."
    )
    parser.add_argument(
        "--skip-rollouts", action="store_true",
        help="Skip Phase 1 (rollout collection). Requires --trajectories-file.",
    )
    parser.add_argument(
        "--trajectories-file", type=Path, default=None,
        help="Path to a previously saved tau2 simulation JSON.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip Phase 2 (RL training). Requires --model-output-dir.",
    )
    parser.add_argument(
        "--model-output-dir", type=Path,
        default=Path("output/airline_rl_tuned"),
        help="Directory to save (or load) the tuned model. Default: output/airline_rl_tuned",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip Phase 3 (post-RL evaluation).",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="HuggingFace Hub repo ID to push the tuned model to (e.g. username/model-name). "
             "Requires HF_TOKEN env var.",
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Disable 4-bit quantisation (use float16 instead). "
             "Only needed if bitsandbytes is unavailable.",
    )
    args = parser.parse_args()

    ts = int(time.time())
    rollout_file = args.trajectories_file or (
        SIM_DIR / f"airline_ms_train_rollouts_{ts}.json"
    )
    post_eval_file = SIM_DIR / f"airline_ms_test_post_rl_{ts}.json"

    # Phase 1 — Rollout collection
    if not args.skip_rollouts:
        collect_rollouts(rollout_file)
    elif not rollout_file.exists():
        parser.error(f"--trajectories-file {rollout_file} does not exist.")

    # Phase 2 — RL training
    if not args.skip_train:
        run_rl_training(
            trajectories_file=rollout_file,
            output_dir=args.model_output_dir,
            push_to_hub=args.push_to_hub,
            quantize_4bit=not args.no_quantize,
        )
    elif not args.model_output_dir.exists():
        parser.error(
            f"--skip-train requires the model to already exist at "
            f"--model-output-dir {args.model_output_dir}."
        )

    # Phase 3 — Post-RL evaluation
    if not args.skip_eval:
        # Use Hub model ID if pushed, otherwise use local path
        eval_model = args.push_to_hub or str(args.model_output_dir)
        run_post_eval(eval_model, post_eval_file)

    # Summary
    print_comparison(rollout_file, post_eval_file if not args.skip_eval else None)
    logger.info("Done.")


if __name__ == "__main__":
    main()
