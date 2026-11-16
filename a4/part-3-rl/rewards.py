"""Reward components for GSM8K RL (Part 4 hook).

The training reward is the SUM of the components named in the RL_REWARD environment variable,
e.g. RL_REWARD="strict" (default, identical to stock nanochat) or RL_REWARD="strict,my_format".
A component can carry a weight: RL_REWARD="strict,my_format:0.2".

Each component is a function
    fn(conversation, response, ctx) -> float
where `conversation` is the GSM8K conversation (question + reference solution), `response` is the
sampled completion text, and `ctx` holds values computed once per sample:
    ctx["correct"]   strict 0/1 correctness (the stock scorer: '#### <number>' must match)
    ctx["pred"]      the number after '####' in the response, or None
    ctx["ref"]       the reference answer string
    ctx["ref_solution_parts"]  the reference solution as nanochat parts (text / python / python_output)

Only the TRAINING reward changes. In-training pass@k and the final chat_eval stay strict
(GSM8K.evaluate is never modified), so every run stays comparable to run #1 and to Karpathy.
The rollout logs record both `correct` (strict) and `reward` (the shaped value), plus each component.

To add a reward: write a function below and register it in REWARDS. Keep it cheap: it runs on every
one of the ~120k training samples, on the training process.
"""

REWARDS = {}


def register(name):
    def deco(fn):
        REWARDS[name] = fn
        return fn
    return deco


@register("strict")
def strict(conversation, response, ctx):
    """Stock nanochat reward: 1.0 if the '####' answer matches the reference, else 0.0."""
    return float(ctx["correct"])


# Example of the function shape only (registered so the hook can be tested; not a Part 4 design):
@register("example_has_marker")
def example_has_marker(conversation, response, ctx):
    """1.0 if the response contains a parseable '#### <number>' answer, else 0.0."""
    return float(ctx["pred"] is not None)


def parse_spec(spec):
    """'strict,foo:0.2' -> [('strict', 1.0), ('foo', 0.2)]; raises on unknown names."""
    parts = []
    for item in (spec or "strict").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        if name not in REWARDS:
            raise ValueError(f"Unknown reward component {name!r}; known: {sorted(REWARDS)}")
        parts.append((name, float(w) if w else 1.0))
    if not parts:
        raise ValueError("RL_REWARD names no components")
    return parts


def compute(spec_parts, conversation, response, ctx):
    """Return (total_reward, {component: weighted value})."""
    vals = {name: w * REWARDS[name](conversation, response, ctx) for name, w in spec_parts}
    return sum(vals.values()), vals
