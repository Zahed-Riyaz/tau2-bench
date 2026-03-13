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
  Phase 3  Evaluate the tuned model on the TEST split by loading it locally
           with transformers — no vllm or external API required.

Quick start (local GPU or Google Colab T4 — free)
-----------
    pip install transformers peft accelerate torch bitsandbytes
    export GROQ_API_KEY=<key>
    python -m tau2.scripts.rl_airline_experiment

    Base model: Qwen/Qwen2.5-7B-Instruct (open, no HF access gate required).
    To use Llama instead: huggingface-cli login, request access at
    https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, then set BASE_MODEL.

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
_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = _REPO_ROOT / "data"          # matches tau2's own DATA_DIR (repo_root/data)
SIM_DIR = DATA_DIR / "simulations"
SIM_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = "airline"
TASK_SET = "airline_multistep"

# Base model options (all open, no HF access gate):
#   Qwen/Qwen2.5-0.5B-Instruct  ~1 GB  — proof-of-pipeline, minimal download
#   Qwen/Qwen2.5-3B-Instruct    ~6 GB  — better quality, still fits on T4
#   Qwen/Qwen2.5-7B-Instruct   ~14 GB  — recommended for real fine-tuning
# For Llama: request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# then run `huggingface-cli login` and set BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1 GB download, no HF gate

# Rollout collection — Groq is fast and free.
ROLLOUT_MODEL = "groq/llama-3.3-70b-versatile"
USER_MODEL = "groq/llama-3.3-70b-versatile"

# RL hyper-params
LORA_RANK = 16
LORA_ALPHA = 32
ADAM_LR = 3e-4
NUM_EPOCHS = 2
REWARD_BASELINE = 0.5   # shift {0,1} reward → {-0.5, +0.5} advantage
MAX_SEQ_LEN = 512       # 2048 on GPU; 512 keeps each CPU step under ~5s for demo runs


# ── Phase 1: Rollout collection ────────────────────────────────────────────────

def collect_rollouts(out_file: Path, agent_llm: str = ROLLOUT_MODEL, split: str = "train") -> Path:
    """Run tau2 on *split* and save trajectories to *out_file*."""
    logger.info(f"Phase 1 — collecting rollouts (split={split}) with agent={agent_llm} …")
    cmd = [
        "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", TASK_SET,
        "--task-split-name", split,
        "--agent-llm", agent_llm,
        "--user-llm", USER_MODEL,
        # tau2 CLI prepends DATA_DIR/simulations/ and appends .json automatically,
        # so we pass only the stem (filename without extension).
        "--save-to", out_file.stem,
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning("tau2 run exited with non-zero status — some tasks may have failed.")
    expected_file = SIM_DIR / f"{out_file.stem}.json"

    if not expected_file.exists():
        # fallback: check alternative tau2 path
        alt = _REPO_ROOT / "data" / "simulations" / f"{out_file.stem}.json"
        if alt.exists():
            expected_file = alt
        else:
            raise FileNotFoundError(
                f"Rollout file not found in expected locations:\n"
                f"{expected_file}\n{alt}"
            )

    logger.info(f"Trajectories saved → {expected_file}")
    return expected_file


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


MIN_REWARD_THRESHOLD = 0.0  # filter_failed: skip trajectories with reward <= this
TOOL_CALL_UPWEIGHT = 3.0    # multiply advantage for tokens inside a tool-call JSON block


def _upweight_tool_tokens(
    resp_tokens: list[int],
    tokenizer,
    base_advantage: float,
    upweight: float = TOOL_CALL_UPWEIGHT,
) -> list[float]:
    """
    Return per-token advantage weights for a single assistant response.

    Tokens that fall inside a JSON tool-call block (between the first '{' and
    matching '}') receive advantage * upweight; all other response tokens receive
    the base advantage. This improves credit assignment: the tool name and
    argument values are the decisions that actually affect task success, while
    surrounding natural language is noise for the reward signal.
    """
    import re
    resp_str = tokenizer.decode(resp_tokens, skip_special_tokens=False)

    # Find byte offsets of all {...} spans that look like tool-call JSON
    tool_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\{[^{}]*\}", resp_str):
        tool_spans.append((m.start(), m.end()))

    weights: list[float] = []
    char_pos = 0
    for tok_id in resp_tokens:
        tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
        tok_end = char_pos + len(tok_str)
        in_tool = any(s <= char_pos < e for s, e in tool_spans)
        weights.append(base_advantage * (upweight if in_tool else 1.0))
        char_pos = tok_end

    return weights


def build_training_tensors(
    simulation_results: list[dict],
    tokenizer,
    baseline: float = REWARD_BASELINE,
    max_len: int = MAX_SEQ_LEN,
    filter_failed: bool = False,
    normalise_advantages: bool = True,
    upweight_tool_calls: bool = True,
) -> list[dict]:
    """
    Convert tau2 SimulationRun dicts → PyTorch training tensors.

    Strategy (REINFORCE with group-normalised advantage):
      advantage = (reward - mean(rewards)) / (std(rewards) + 1e-8)
      Each assistant turn in the episode gets the same advantage as a
      per-token weight. Positive advantage → reinforce those tokens.
      Negative advantage → suppress them.

    normalise_advantages: if True, use group normalisation (GRPO-style) instead
      of a fixed baseline. Keeps advantage scale stable regardless of the
      reward distribution in the current batch — critical when most rewards are
      0 (constant baseline=0.5 would assign large negative advantage to every
      token in every failed episode, overwhelming the few positive signals).

    upweight_tool_calls: if True, multiply the advantage for tokens inside JSON
      tool-call blocks by TOOL_CALL_UPWEIGHT. The tool name and argument values
      are the decisions that actually determine task success; upweighting them
      improves credit assignment without changing the loss function structure.

    filter_failed: if True, skip trajectories with reward <= MIN_REWARD_THRESHOLD.
      Converts REINFORCE into reward-weighted regression — only successful episodes
      contribute gradient. Reduces noise when failed episodes dominate the dataset.

    Returns a list of dicts with keys:
      input_ids  — LongTensor [seq_len]
      weights    — FloatTensor [seq_len]  (0 for prompt, ±advantage for response)
    """
    training_data = []

    # ── Group-normalised advantage (GRPO-style) ───────────────────────────────
    rewards = [sim.get("reward", 0.0) for sim in simulation_results]
    import statistics
    r_mean = statistics.mean(rewards) if rewards else 0.0
    r_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    # Fall back to fixed baseline when std ≈ 0 (all rewards identical, e.g. all-zero
    # baseline run). Normalising by near-zero std would collapse every advantage to 0
    # and produce zero training tensors. Fixed baseline still gives a valid gradient:
    # with all rewards=0 and baseline=0.5, advantage=-0.5 suppresses every action taken.
    use_normalise = normalise_advantages and r_std > 1e-6
    if use_normalise:
        logger.info(
            f"Advantage normalisation (GRPO): mean={r_mean:.3f} std={r_std:.3f} "
            f"across {len(rewards)} trajectories."
        )
        def compute_advantage(r: float) -> float:
            return (r - r_mean) / r_std
    else:
        if normalise_advantages and r_std <= 1e-6:
            logger.warning(
                f"All rewards identical ({r_mean:.3f}) — std≈0, "
                f"falling back to fixed baseline={baseline}."
            )
        def compute_advantage(r: float) -> float:
            return r - baseline

    skipped_failed = 0
    for sim in simulation_results:
        reward: float = sim.get("reward", 0.0)

        if filter_failed and reward <= MIN_REWARD_THRESHOLD:
            skipped_failed += 1
            logger.debug(f"Task {sim.get('task_id')}: reward={reward:.2f} filtered out.")
            continue

        advantage = compute_advantage(reward)
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

            # ── Per-token weights ─────────────────────────────────────────────
            if upweight_tool_calls:
                resp_weights = _upweight_tool_tokens(resp_tokens, tokenizer, advantage)
            else:
                resp_weights = [advantage] * len(resp_tokens)
            weights = [0.0] * len(ctx_tokens) + resp_weights

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
        f"from {len(simulation_results)} trajectories"
        + (f" ({skipped_failed} failed filtered out)." if filter_failed else ".")
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
    filter_failed: bool = False,
    normalise_advantages: bool = True,
    upweight_tool_calls: bool = True,
    max_steps: int | None = None,
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
    training_data = build_training_tensors(
        simulations, tokenizer,
        filter_failed=filter_failed,
        normalise_advantages=normalise_advantages,
        upweight_tool_calls=upweight_tool_calls,
    )

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

    steps_per_epoch = (
        min(max_steps, len(training_data)) if max_steps else len(training_data)
    )
    if max_steps and max_steps < len(training_data):
        logger.info(
            f"--max-steps {max_steps}: capping each epoch at {max_steps}/{len(training_data)} steps."
        )

    model.train()
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} — {steps_per_epoch} steps …")
        total_loss = 0.0

        for step, batch in enumerate(training_data[:steps_per_epoch]):
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
            f"Epoch {epoch + 1} done — avg_loss={total_loss / steps_per_epoch:.4f}"
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

if TRL_OK:
    from tau2.agent.llm_agent import LLMAgent, LLMAgentState
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall

    class LocalHFAgent(LLMAgent):
        """
        Drop-in replacement for LLMAgent that calls a local HuggingFace model
        instead of going through litellm.  Used for Phase 3 post-RL evaluation
        so no external API or vllm server is needed.
        """

        def __init__(self, hf_model, tokenizer, tools, domain_policy):
            super().__init__(tools=tools, domain_policy=domain_policy, llm="local")
            self._hf_model = hf_model
            self._hf_tokenizer = tokenizer

        def generate_next_message(
            self, message, state: LLMAgentState
        ) -> tuple[AssistantMessage, LLMAgentState]:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            messages = state.system_messages + state.messages
            assistant_message = _local_hf_generate(
                self._hf_model, self._hf_tokenizer, messages, self.tools
            )
            state.messages.append(assistant_message)
            return assistant_message, state


def _parse_local_tool_calls(text: str) -> list:
    """
    Parse tool calls from Qwen2.5's <tool_call>...</tool_call> output format.
    Returns a list of ToolCall objects (empty list if none found).
    """
    import re
    import uuid

    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            data = json.loads(m.group(1))
            calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=data["name"],
                    arguments=data.get("arguments", {}),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")
    return calls


def _local_hf_generate(hf_model, tokenizer, messages, tools) -> "AssistantMessage":
    """
    Run one forward pass on the local HF model and return an AssistantMessage.
    Formats input with apply_chat_template (supports tool schemas for Qwen2.5),
    generates greedily, then parses any <tool_call> blocks in the output.
    """
    from tau2.utils.llm_utils import to_litellm_messages

    tools_schema = [t.openai_schema for t in tools] if tools else None
    oai_msgs = to_litellm_messages(messages)

    input_text = tokenizer.apply_chat_template(
        oai_msgs,
        tools=tools_schema,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(hf_model.device)

    with torch.no_grad():
        output_ids = hf_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    tool_calls = _parse_local_tool_calls(response_text) or None
    # When the model makes tool calls the content should be None; otherwise it's text.
    content = None if tool_calls else response_text

    return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)


def run_post_eval_local(model_dir: Path, out_file: Path) -> None:
    """
    Evaluate the locally saved tuned model on the TEST split.

    Loads model weights from *model_dir* with transformers, plugs them into a
    LocalHFAgent, and runs tau2's orchestrator + evaluator directly — no vllm,
    no external API required.  Results are saved to *out_file*.
    """
    if not TRL_OK:
        raise ImportError(
            "transformers/peft/torch not installed.\n"
            "Run: pip install transformers peft accelerate torch"
        )

    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.run import load_tasks

    logger.info(f"Phase 3 — post-RL evaluation (local model at {model_dir}) …")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map="auto",
        dtype=torch.float16,
    )
    hf_model.eval()
    logger.info("Model loaded.")

    tasks = load_tasks(TASK_SET, task_split_name="test")
    env_constructor = registry.get_env_constructor(DOMAIN)
    UserConstructor = registry.get_user_constructor("user_simulator")

    results = []
    for task in tasks:
        logger.info(f"  Evaluating task {task.id} …")
        try:
            environment = env_constructor()
            agent = LocalHFAgent(
                hf_model=hf_model,
                tokenizer=tokenizer,
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
            )
            try:
                user_tools = environment.get_user_tools()
            except Exception:
                user_tools = None
            user = UserConstructor(
                tools=user_tools,
                instructions=str(task.user_scenario),
                llm=USER_MODEL,
                llm_args={},
            )
            orchestrator = Orchestrator(
                domain=DOMAIN,
                agent=agent,
                user=user,
                environment=environment,
                task=task,
                max_steps=100,
                max_errors=10,
                seed=42,
            )
            simulation = orchestrator.run()
            reward_info = evaluate_simulation(
                domain=DOMAIN,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
            )
            reward = reward_info.reward
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            reward = 0.0

        results.append({"task_id": task.id, "reward": reward})
        logger.info(f"  Task {task.id}: reward={reward:.4f}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"simulations": results}, indent=2))
    logger.info(f"Post-RL results saved → {out_file}")


def run_post_eval(model_path: str, out_file: Path) -> None:
    """
    Evaluate the tuned model on the TEST split.

    If *model_path* is a local directory (the default after Phase 2), runs
    evaluation directly using the local HF model — no vllm or external API needed.

    If *model_path* is a HuggingFace Hub repo ID (set via --push-to-hub), falls
    back to calling the HF Inference Providers API via litellm.
    """
    if Path(model_path).exists() and Path(model_path).is_dir():
        run_post_eval_local(Path(model_path), out_file)
        return

    # Hub model ID path — requires HF_TOKEN and an approved provider
    logger.info(f"Phase 3 — post-RL evaluation on test split (Hub model={model_path}) …")
    agent_llm = f"huggingface/{model_path}"

    cmd = [
        "tau2", "run",
        "--domain", DOMAIN,
        "--task-set-name", TASK_SET,
        "--task-split-name", "test",
        "--agent-llm", agent_llm,
        "--user-llm", USER_MODEL,
        "--save-to", out_file.stem,
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
        result = {}
        for sim in data.get("simulations", []):
            tid = sim.get("task_id")
            if tid is None:
                continue
            # Flat format (local eval): {"task_id": ..., "reward": ...}
            if "reward" in sim:
                result[tid] = float(sim["reward"])
            # Nested format (tau2 CLI Results): reward_info.reward
            elif sim.get("reward_info"):
                result[tid] = float(sim["reward_info"].get("reward", float("nan")))
            else:
                result[tid] = float("nan")
        return result

    before = _load(before_file)
    after = _load(after_file)
    all_ids = sorted(set(before) | set(after))

    print("\n" + "=" * 55)
    print(f"{'Task':<12}  {'Before (baseline)':>18}  {'After (RL)':>12}")
    print("-" * 55)
    for tid in all_ids:
        b = before.get(tid)
        a = after.get(tid)
        b_str = f"{b:.4f}" if b is not None else "N/A"
        a_str = f"{a:.4f}" if a is not None else "N/A (run --push-to-hub to enable)"
        arrow = "→"
        if b is not None and a is not None:
            arrow = "↑" if a > b else ("↓" if a < b else "→")
        print(f"{tid:<12}  {b_str:>18}  {arrow} {a_str}")
    print("=" * 55)

    if before:
        avg_b = sum(before.values()) / len(before)
        avg_b_str = f"{avg_b:.4f}"
        if after:
            avg_a = sum(after.values()) / len(after)
            print(f"{'AVERAGE':<12}  {avg_b_str:>18}  → {avg_a:.4f}")
        else:
            print(f"{'AVERAGE':<12}  {avg_b_str:>18}  → N/A")
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
    parser.add_argument(
        "--filter-failed", action="store_true",
        help="Skip reward=0 trajectories during training (reward-weighted regression). "
             "Useful when failed episodes dominate the dataset.",
    )
    parser.add_argument(
        "--rl-iterations", type=int, default=1,
        help="Number of on-policy RL iterations. After the first iteration the trained "
             "model generates rollouts for the next iteration. Default: 1.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Cap each epoch at this many gradient steps. "
             "Use for quick demo runs on CPU (e.g. --max-steps 10).",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Disable group-normalised advantages (GRPO-style). "
             "Falls back to fixed baseline=0.5. Use if batch size is too small for stable stats.",
    )
    parser.add_argument(
        "--no-upweight", action="store_true",
        help="Disable tool-call token upweighting. "
             "All response tokens get equal advantage weight.",
    )
    args = parser.parse_args()

    ts = int(time.time())
    rollout_file = args.trajectories_file or (
        SIM_DIR / f"airline_ms_train_rollouts_{ts}.json"
    )
    baseline_file = SIM_DIR / f"airline_ms_test_baseline_{ts}.json"
    post_eval_file = SIM_DIR / f"airline_ms_test_post_rl_{ts}.json"

    # Phase 0 — Collect test-split baseline with the rollout model (Groq).
    # This gives the "Before" column in the comparison table using the SAME
    # test tasks that post-eval will score. Skipped when --skip-rollouts is set
    # (assume the user already has a baseline or is in a quick-demo mode).
    if not args.skip_rollouts:
        logger.info("Phase 0 — collecting test-split baseline (Groq) for comparison table …")
        collect_rollouts(baseline_file, agent_llm=ROLLOUT_MODEL, split="test")
    else:
        logger.info("Phase 0 skipped (--skip-rollouts). Before column will show N/A.")

    # ── On-policy iteration loop ──────────────────────────────────────────────
    # Iteration 0: rollout agent = ROLLOUT_MODEL (external model)
    # Iteration k>0: rollout agent = trained model from previous iteration
    # This converts offline REINFORCE into an on-policy improvement loop.

    for iteration in range(args.rl_iterations):
        is_first = (iteration == 0)
        iter_rollout_file = (
            rollout_file if is_first
            else SIM_DIR / f"airline_ms_train_iter{iteration}_{ts}.json"
        )
        iter_model_dir = (
            args.model_output_dir if args.rl_iterations == 1
            else args.model_output_dir.parent / f"{args.model_output_dir.name}_iter{iteration}"
        )

        if iteration > 0:
            logger.info(
                f"On-policy iteration {iteration}/{args.rl_iterations - 1} — "
                f"collecting rollouts with trained model …"
            )

        # Phase 1 — Rollout collection
        if not args.skip_rollouts or not is_first:
            rollout_agent = (
                ROLLOUT_MODEL if is_first
                else f"huggingface/{args.model_output_dir}"
            )
            collect_rollouts(iter_rollout_file, agent_llm=rollout_agent)
        elif not iter_rollout_file.exists():
            parser.error(f"--trajectories-file {iter_rollout_file} does not exist.")

        # Phase 2 — RL training
        if not args.skip_train:
            run_rl_training(
                trajectories_file=iter_rollout_file,
                output_dir=iter_model_dir,
                push_to_hub=args.push_to_hub if iteration == args.rl_iterations - 1 else None,
                quantize_4bit=not args.no_quantize,
                filter_failed=args.filter_failed,
                normalise_advantages=not args.no_normalise,
                upweight_tool_calls=not args.no_upweight,
                max_steps=args.max_steps,
            )
            # Point subsequent iterations at the freshly trained model
            args.model_output_dir = iter_model_dir
        elif not iter_model_dir.exists():
            parser.error(
                f"--skip-train requires the model to already exist at "
                f"--model-output-dir {iter_model_dir}."
            )

    # Phase 3 — Post-RL evaluation
    if not args.skip_eval:
        # Use Hub model ID if pushed, otherwise use local path
        eval_model = args.push_to_hub or str(args.model_output_dir)
        run_post_eval(eval_model, post_eval_file)

    # Summary — compare test-split baseline vs post-RL test results
    before_file = baseline_file if baseline_file.exists() else None
    print_comparison(before_file, post_eval_file if not args.skip_eval else None)
    logger.info("Done.")


if __name__ == "__main__":
    main()
