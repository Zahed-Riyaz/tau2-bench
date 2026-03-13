"""
RL fine-tuning experiment for airline_multistep.

Pipeline
--------
  Phase 0  Score the test split with Groq — this is the "Before" baseline.
  Phase 1  Collect full conversation trajectories on the train split via Groq.
  Phase 2  REINFORCE + LoRA: fine-tune a small open model on those trajectories.
  Phase 3  Push the tuned model to HuggingFace Hub, score the test split again,
           print a Before / After reward table.

Quick start (requires a GPU — free Colab T4 works)
-----------
    pip install transformers peft accelerate torch bitsandbytes
    export GROQ_API_KEY=<key>
    python -m tau2.scripts.rl_airline_experiment --push-to-hub <hf-user>/airline-rl

Skip rollout collection if you already have a simulation file:
    python -m tau2.scripts.rl_airline_experiment \\
        --skip-rollouts \\
        --trajectories-file data/tau2/simulations/<run>.json \\
        --push-to-hub <hf-user>/airline-rl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from loguru import logger

try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRL_OK = True
except ImportError:
    TRL_OK = False
    logger.warning(
        "transformers/peft/torch not installed — Phases 2 and 3 will fail.\n"
        "Run: pip install transformers peft accelerate torch bitsandbytes"
    )

# ── Paths ──────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = _REPO_ROOT / "data" / "simulations"
SIM_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

DOMAIN        = "airline"
TASK_SET      = "airline_multistep"

# Base model — open, no HuggingFace access gate required.
# Swap to Qwen/Qwen2.5-7B-Instruct for real results (needs a GPU with ~14 GB VRAM).
BASE_MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"

# Groq is used for rollout collection and the user simulator (fast, free).
ROLLOUT_MODEL = "groq/llama-3.3-70b-versatile"
USER_MODEL    = "groq/llama-3.3-70b-versatile"

# RL hyper-parameters
LORA_RANK      = 16       # number of LoRA low-rank dimensions
LORA_ALPHA     = 32       # LoRA scaling factor
ADAM_LR        = 3e-4
NUM_EPOCHS     = 2
REWARD_BASELINE = 0.5     # shift {0,1} reward → {-0.5, +0.5} advantage
MAX_SEQ_LEN    = 512      # truncate sequences to this length


# ── Phase 0 & 1: Rollout collection ───────────────────────────────────────────

def collect_rollouts(out_file: Path, split: str) -> None:
    """Run tau2 on *split* and save the full conversation trajectories to *out_file*."""
    logger.info(f"Collecting rollouts — split={split}, saving to {out_file.stem} …")
    result = subprocess.run([
        "tau2", "run",
        "--domain",          DOMAIN,
        "--task-set-name",   TASK_SET,
        "--task-split-name", split,
        "--agent-llm",       ROLLOUT_MODEL,
        "--user-llm",        USER_MODEL,
        "--save-to",         out_file.stem,  # tau2 appends .json automatically
    ])
    if result.returncode != 0:
        logger.warning("tau2 run exited non-zero — some tasks may have failed.")

    # tau2 saves into its own data/simulations/ directory; find the file.
    tau2_sim_dir = _REPO_ROOT / "data" / "tau2" / "simulations"
    candidate = tau2_sim_dir / f"{out_file.stem}.json"
    if candidate.exists() and candidate != out_file:
        candidate.rename(out_file)


# ── Phase 2: REINFORCE + LoRA fine-tuning ─────────────────────────────────────

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """
    Convert tau2 message dicts (as stored in the simulation JSON) to the
    OpenAI chat format that HuggingFace tokenizers expect for apply_chat_template.
    """
    out = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            entry: dict = {"role": "assistant", "content": m.get("content") or ""}
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
    Tokenise one assistant turn.

    Returns context_tokens (prompt) and response_tokens (completion) separately
    so we can assign weight=0 to the prompt and weight=advantage to the response.
    """
    ctx_str  = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    full_str = tokenizer.apply_chat_template(context + [response], tokenize=False, add_generation_prompt=False)
    ctx_tokens  = tokenizer.encode(ctx_str,  add_special_tokens=False)
    full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
    resp_tokens = full_tokens[len(ctx_tokens):]
    return ctx_tokens, resp_tokens


def build_training_tensors(simulations: list[dict], tokenizer) -> list[dict]:
    """
    Convert tau2 simulation runs into PyTorch training tensors.

    REINFORCE strategy:
      advantage = reward - REWARD_BASELINE
        → positive advantage: reinforce these tokens (agent did well)
        → negative advantage: suppress these tokens (agent did poorly)

    Each assistant turn in the conversation becomes one training sample.
    Prompt tokens get weight=0 (we don't train the model to predict context).
    Response tokens get weight=advantage.
    """
    training_data = []
    for sim in simulations:
        # Handle both flat {"reward": x} and nested {"reward_info": {"reward": x}} formats
        reward_info = sim.get("reward_info") or {}
        reward = float(sim.get("reward") or reward_info.get("reward") or 0.0)
        advantage = reward - REWARD_BASELINE

        if abs(advantage) < 1e-6:
            continue  # reward exactly at baseline → zero gradient, skip

        oai_messages = _to_openai_messages(sim.get("messages", []))

        for i, msg in enumerate(oai_messages):
            if msg["role"] != "assistant":
                continue
            context = oai_messages[:i]
            if not context:
                continue  # need at least one prior message

            try:
                ctx_tokens, resp_tokens = _encode_turn(tokenizer, context, msg)
            except Exception as e:
                logger.warning(f"Tokenisation failed at turn {i}: {e}")
                continue

            if not resp_tokens:
                continue

            # Concatenate and truncate from the right (keep recent context)
            all_tokens = (ctx_tokens + resp_tokens)[-MAX_SEQ_LEN:]
            # Weights: 0 for prompt tokens, advantage for response tokens
            weights    = ([0.0] * len(ctx_tokens) + [advantage] * len(resp_tokens))[-MAX_SEQ_LEN:]

            training_data.append({
                "input_ids": torch.tensor(all_tokens, dtype=torch.long),
                "weights":   torch.tensor(weights,    dtype=torch.float32),
            })

    logger.info(f"Built {len(training_data)} training tensors from {len(simulations)} episodes.")
    return training_data


def setup_model(quantize_4bit: bool = True):
    """
    Load BASE_MODEL and attach LoRA adapters.

    LoRA (Low-Rank Adaptation) freezes the original model weights and adds a small
    set of trainable rank-decomposition matrices (~0.1% of parameters). This lets
    us fine-tune a 7B model on a single GPU with 16 GB VRAM.

    quantize_4bit=True loads weights in 4-bit precision via bitsandbytes,
    reducing VRAM usage from ~14 GB to ~5 GB for a 7B model.
    """
    if not TRL_OK:
        raise ImportError("Run: pip install transformers peft accelerate torch bitsandbytes")

    logger.info(f"Loading {BASE_MODEL} …")
    load_kwargs: dict = {"device_map": "auto"}
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

    model     = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
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
    return model, tokenizer


def run_rl_training(
    trajectories_file: Path,
    output_dir: Path,
    push_to_hub: str | None = None,
    quantize_4bit: bool = True,
) -> None:
    """
    REINFORCE fine-tune BASE_MODEL on the collected trajectories.

    Loss = -mean( log_prob(token) * advantage )
    Minimising this loss maximises the log-probability of tokens from successful
    episodes and minimises it for tokens from failed episodes.
    """
    logger.info("Phase 2 — REINFORCE + LoRA fine-tuning …")

    data        = json.loads(trajectories_file.read_text())
    simulations = data if isinstance(data, list) else data.get("simulations", [])
    logger.info(f"Loaded {len(simulations)} simulation runs.")

    model, tokenizer = setup_model(quantize_4bit=quantize_4bit)
    training_data    = build_training_tensors(simulations, tokenizer)

    if not training_data:
        raise ValueError("No training tensors — check that trajectories_file has valid runs.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ADAM_LR,
        weight_decay=0.01,
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for step, sample in enumerate(training_data):
            input_ids = sample["input_ids"].unsqueeze(0).to(model.device)  # [1, seq]
            weights   = sample["weights"].unsqueeze(0).to(model.device)    # [1, seq]

            logits = model(input_ids=input_ids).logits                     # [1, seq, vocab]

            # Shift: predict token[t+1] given tokens[0..t]
            shift_logits  = logits[:, :-1, :].contiguous()                 # [1, seq-1, vocab]
            shift_labels  = input_ids[:, 1:].contiguous()                  # [1, seq-1]
            shift_weights = weights[:, 1:].contiguous()                    # [1, seq-1]

            # Cross-entropy loss per token, then weight by advantage
            token_losses = nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(1, -1)                                                  # [1, seq-1]

            loss = (token_losses * shift_weights).sum() / (shift_weights.abs().sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % 5 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{NUM_EPOCHS}  step {step+1}/{len(training_data)}"
                    f"  avg_loss={total_loss/(step+1):.4f}"
                )

        logger.info(f"Epoch {epoch+1} done — avg_loss={total_loss/len(training_data):.4f}")

    # Merge LoRA weights back into the base model weights and save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved → {output_dir}")

    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub → {push_to_hub} …")
        merged.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)
        logger.info(f"Available at: https://huggingface.co/{push_to_hub}")


# ── Phase 3: Post-RL evaluation ────────────────────────────────────────────────

def run_post_eval(hub_model_id: str, out_file: Path) -> None:
    """
    Score the test split using the tuned model served from HuggingFace Hub.
    tau2 routes huggingface/<model-id> through litellm to the HF Inference API.
    """
    logger.info(f"Phase 3 — evaluating {hub_model_id} on test split …")
    subprocess.run([
        "tau2", "run",
        "--domain",          DOMAIN,
        "--task-set-name",   TASK_SET,
        "--task-split-name", "test",
        "--agent-llm",       f"huggingface/{hub_model_id}",
        "--user-llm",        USER_MODEL,
        "--save-to",         out_file.stem,
    ])


# ── Comparison table ───────────────────────────────────────────────────────────

def print_comparison(before_file: Path | None, after_file: Path | None) -> None:
    """Print a Before / After reward table for the test split."""

    def _load(path: Path | None) -> dict[str, float]:
        if path is None or not path.exists():
            return {}
        data = json.loads(path.read_text())
        sims = data if isinstance(data, list) else data.get("simulations", [])
        out  = {}
        for sim in sims:
            tid = sim.get("task_id")
            if not tid:
                continue
            r = sim.get("reward") or (sim.get("reward_info") or {}).get("reward")
            if r is not None:
                out[tid] = float(r)
        return out

    before = _load(before_file)
    after  = _load(after_file)
    all_ids = sorted(set(before) | set(after))

    print("\n" + "=" * 57)
    print(f"{'Task':<14} {'Before':>12}    {'After':>12}")
    print("-" * 57)
    bvals, avals = [], []
    for tid in all_ids:
        b, a = before.get(tid), after.get(tid)
        b_s  = f"{b:.4f}" if b is not None else "N/A"
        a_s  = f"{a:.4f}" if a is not None else "N/A (use --push-to-hub)"
        arrow = " "
        if b is not None and a is not None:
            arrow = "↑" if a > b else ("↓" if a < b else "→")
            bvals.append(b); avals.append(a)
        print(f"{tid:<14} {b_s:>12} {arrow:>4} {a_s}")
    print("=" * 57)
    if bvals:
        ab, aa = sum(bvals)/len(bvals), sum(avals)/len(avals)
        arrow = "↑" if aa > ab else ("↓" if aa < ab else "→")
        print(f"{'AVERAGE':<14} {ab:>12.4f} {arrow:>4} {aa:.4f}")
    print()


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REINFORCE RL fine-tuning for airline_multistep."
    )
    parser.add_argument("--skip-rollouts", action="store_true",
                        help="Skip Phases 0 & 1. Requires --trajectories-file.")
    parser.add_argument("--trajectories-file", type=Path, default=None,
                        help="Path to an existing tau2 simulation JSON (skips Phase 1).")
    parser.add_argument("--model-output-dir", type=Path,
                        default=Path("output/airline_rl_tuned"),
                        help="Where to save the tuned model weights.")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF Hub repo id (e.g. username/airline-rl). "
                             "Required to run Phase 3 evaluation.")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Use float16 instead of 4-bit quantisation.")
    args = parser.parse_args()

    ts = int(time.time())
    baseline_file  = SIM_DIR / f"airline_ms_baseline_test_{ts}.json"
    rollout_file   = args.trajectories_file or (SIM_DIR / f"airline_ms_train_{ts}.json")
    post_eval_file = SIM_DIR / f"airline_ms_post_eval_{ts}.json"

    # Phase 0 — baseline on test split (always runs unless --skip-rollouts)
    if not args.skip_rollouts:
        logger.info("Phase 0 — baseline (test split, Groq) …")
        collect_rollouts(baseline_file, split="test")

        # Phase 1 — rollouts on train split
        logger.info("Phase 1 — rollout collection (train split, Groq) …")
        collect_rollouts(rollout_file, split="train")
    else:
        if not rollout_file.exists():
            parser.error(f"--trajectories-file {rollout_file} does not exist.")
        logger.info("Phases 0 & 1 skipped (--skip-rollouts).")

    # Phase 2 — RL training
    run_rl_training(
        trajectories_file=rollout_file,
        output_dir=args.model_output_dir,
        push_to_hub=args.push_to_hub,
        quantize_4bit=not args.no_quantize,
    )

    # Phase 3 — post-RL evaluation (only if model was pushed to Hub)
    if args.push_to_hub:
        run_post_eval(args.push_to_hub, post_eval_file)
    else:
        logger.info(
            "Phase 3 skipped — pass --push-to-hub <hf-user>/<repo> to evaluate "
            "the tuned model and fill in the After column."
        )

    print_comparison(
        before_file=baseline_file if not args.skip_rollouts else None,
        after_file=post_eval_file if args.push_to_hub else None,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
