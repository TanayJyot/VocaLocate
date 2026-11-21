"""
On-policy distillation on GSM8K (A4 Part 5).

A copy of nanochat's scripts/chat_rl.py with one change: the per-token advantage.
Sampling, data order, batch sizes, LR schedule, pass@k evaluation and checkpointing
are unchanged, so runs are directly comparable with the Part 3 / Part 4 RL runs.

For every sampled completion y and every generated token y_t:

    A_t = reward_coef * (r - mean_group(r))                                  # chat_rl's GRPO-ish term
        + distill_coef * (log p_teacher(y_t | ctx_T, y<t) - log p_student(y_t | x, y<t))   # -reverse KL

with the teacher term under stop-gradient and gamma = 0 (each token is credited only
with its own log-ratio), as in Thinking Machines' on-policy distillation (2025).
The loss is chat_rl's token-level policy gradient, -sum_t A_t log p_student(y_t) / N_tokens,
so with reward_coef=0, distill_coef=1 its gradient is that of the per-token reverse KL
KL(student || teacher) estimated on the student's own samples.

Teachers (--teacher):
  checkpoint  a separate frozen model loaded from --teacher-dir/--teacher-step, seeing the
              same prompt as the student (classic on-policy distillation; D00, D02).
  self        SDPO-style self-teacher (Hubotter et al., 2026): a copy of the student whose
              prompt also contains the GSM8K reference solution. --teacher-ema sets how fast
              the copy tracks the student (0 = frozen at the initial model). The paper uses a
              successful sibling rollout as the "solution"; our SFT start solves ~3% of GSM8K, so
              most groups would have none, and we use the reference solution instead (D01).

Run through a4/part-3-rl/rl_rollout_logger.py (RL_SCRIPT=chat_distill) for rollout logging,
smoke-test truncation and preemption resume; see a4/part-5-distill/README.md.
"""

import argparse
import copy
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model, build_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Distillation
parser.add_argument("--teacher", type=str, default="checkpoint", choices=["checkpoint", "self"], help="checkpoint = separate frozen model, self = SDPO-style self-teacher with the reference solution in context")
parser.add_argument("--teacher-dir", type=str, default=None, help="checkpoint directory of the teacher (--teacher=checkpoint)")
parser.add_argument("--teacher-step", type=int, default=None, help="checkpoint step of the teacher (--teacher=checkpoint)")
parser.add_argument("--teacher-ema", type=float, default=0.0, help="--teacher=self: EMA rate toward the student after each step (0 = frozen initial model)")
parser.add_argument("--distill-coef", type=float, default=1.0, help="weight of the per-token teacher log-ratio in the advantage")
parser.add_argument("--reward-coef", type=float, default=0.0, help="weight of chat_rl's (r - mean r) advantage (0 = pure distillation)")
parser.add_argument("--adv-clip", type=float, default=0.0, help="clip the per-token distillation term to [-c, c] (0 = no clipping)")
args = parser.parse_args()
user_config = vars(args).copy()
if args.teacher == "checkpoint" and (args.teacher_dir is None or args.teacher_step is None):
    parser.error("--teacher=checkpoint needs --teacher-dir and --teacher-step")
if args.teacher == "self" and args.teacher_dir is not None:
    parser.error("--teacher=self copies the student; do not pass --teacher-dir")
if not 0.0 <= args.teacher_ema <= 1.0:
    parser.error("--teacher-ema must be in [0, 1]")
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts

# Init the teacher (never trained: no optimizer, no gradients)
if args.teacher == "checkpoint":
    teacher, teacher_tokenizer, _ = build_model(args.teacher_dir, args.teacher_step, device, phase="eval")
    assert teacher_tokenizer.get_vocab_size() == tokenizer.get_vocab_size(), "teacher and student must share a tokenizer"
    print0(f"Teacher: {args.teacher_dir} step {args.teacher_step}")
else:
    teacher = copy.deepcopy(model)
    print0(f"Teacher: self (reference solution in context), EMA rate {args.teacher_ema}")
teacher.eval()
teacher.requires_grad_(False)

def reference_solution_text(conversation):
    """The GSM8K reference solution as plain text, with the <<expr=result>> calculator annotations restored."""
    out = []
    for part in conversation["messages"][-1]["content"]:
        if part["type"] == "python":
            out.append(f"<<{part['text']}=")
        elif part["type"] == "python_output":
            out.append(f"{part['text']}>>")
        else:
            out.append(part["text"])
    return "".join(out)

def self_teacher_conversation(conversation):
    """SDPO reprompt: the question, a correct solution, then the instruction to solve the question again.
    The assistant turn (the student's own sample) is appended token-for-token by the caller."""
    question = conversation["messages"][0]["content"]
    prompt = (f"{question}\n\nCorrect solution:\n{reference_solution_text(conversation)}\n\n"
              "Correctly solve the original question.")
    return {"messages": [{"role": "user", "content": prompt}, conversation["messages"][-1]]}

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # must make sure to change the seed for each sampling step
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # Teacher view of the same samples. The checkpoint teacher sees exactly the student's tokens.
        # The self-teacher sees a longer prompt followed by the same completion tokens; `offset` maps
        # student position j to teacher position j + offset.
        if args.teacher == "self":
            teacher_prefix = tokenizer.render_for_completion(self_teacher_conversation(conversation))
            offset = len(teacher_prefix) - prefix_length
            teacher_ids = torch.tensor([teacher_prefix + seq[prefix_length:] for seq in padded_generated_token_sequences],
                                       dtype=torch.long, device=device)
            teacher_inputs = teacher_ids[:, :-1]
            teacher_targets = torch.full_like(teacher_inputs, -1)
            teacher_targets[:, offset:] = targets # prompt positions of `targets` are already -1
        else:
            offset, teacher_inputs, teacher_targets = 0, inputs, targets
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages, teacher_inputs, teacher_targets, offset

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []
    # Distillation statistics, summed over valid (trained) tokens on this rank
    kl_stats = torch.zeros(4, device=device) # [sum of (log p_S - log p_T), sum of its square, sum of clipped-token count, token count]
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, teacher_inputs_all, teacher_targets_all, offset = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Teacher log probabilities of the same sampled tokens (stop-gradient), aligned to student positions
            with torch.no_grad():
                teacher_inputs = teacher_inputs_all[b0:b1]
                teacher_logp = -teacher(teacher_inputs, teacher_targets_all[b0:b1], loss_reduction='none').view_as(teacher_inputs)
                teacher_logp = teacher_logp[:, offset:]
            valid = targets >= 0
            # Per-token advantage: sequence-level reward term + per-token -reverse-KL term (gamma = 0)
            log_ratio = (teacher_logp - logp.detach()) * valid # log p_T - log p_S, 0 on prompt/tool/padding tokens
            distill_adv = log_ratio.clamp(-args.adv_clip, args.adv_clip) if args.adv_clip > 0 else log_ratio
            token_advantages = args.reward_coef * advantages.unsqueeze(-1) + args.distill_coef * distill_adv
            kl_stats += torch.stack([-log_ratio.sum(), log_ratio.square().sum(),
                                     ((log_ratio != distill_adv) & valid).sum().float(), valid.sum().float()])
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * token_advantages).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = valid.sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    if ddp:
        dist.all_reduce(kl_stats, op=dist.ReduceOp.SUM)
    num_tokens = kl_stats[3].clamp(min=1)
    reverse_kl = (kl_stats[0] / num_tokens).item() # per-token reverse KL estimate, E[log p_S - log p_T]
    log_ratio_std = (kl_stats[1] / num_tokens - (kl_stats[0] / num_tokens) ** 2).clamp(min=0).sqrt().item()
    clip_frac = (kl_stats[2] / num_tokens).item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f} | "
           f"Reverse KL: {reverse_kl:.4f} | Log-ratio std: {log_ratio_std:.4f} | Clip frac: {clip_frac:.4f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "reverse_kl": reverse_kl,
        "log_ratio_std": log_ratio_std,
        "clip_frac": clip_frac,
    })

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    # SDPO teacher regularization: the self-teacher slowly tracks the student (EMA); 0 keeps it frozen
    if args.teacher == "self" and args.teacher_ema > 0:
        with torch.no_grad():
            for teacher_param, param in zip(teacher.parameters(), model.parameters()):
                teacher_param.lerp_(param, args.teacher_ema)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        default_tag = f"d{depth}" + (f"-{model.config.mlp_activation}" if model.config.mlp_activation != "relu2" else "")
        if model.config.diff_v2:
            default_tag += "-diff-v2"
        if model.config.depth_attention:
            default_tag += "-depth-attention"
            if model.config.depth_attention_stride is not None:
                default_tag += f"-s{model.config.depth_attention_stride}"
        if model.config.mgvr:
            default_tag += "-mgvr"
        output_dirname = args.model_tag if args.model_tag else default_tag # base the model tag on the depth and activation of the base model
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = dict(model.config.__dict__) # copy so filtering does not mutate the live config
        if model.config.mlp_activation == "relu2":
            model_config_kwargs.pop("mlp_activation")
        if not model.config.diff_v2:
            model_config_kwargs.pop("diff_v2")
        if not model.config.depth_attention:
            model_config_kwargs.pop("depth_attention")
            model_config_kwargs.pop("depth_attention_stride")
        if not model.config.mgvr:
            model_config_kwargs.pop("mgvr")
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

wandb_run.finish() # wandb run finish
compute_cleanup()
