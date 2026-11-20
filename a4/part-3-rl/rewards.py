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

import ast
from fractions import Fraction
from functools import lru_cache
import math
import re


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


NUMBER_BODY = r"(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
ANSWER_LINE = re.compile(r"#### -?" + NUMBER_BODY)
TOOL_CALL = re.compile(
    r"<\|python_start\|>(.*?)<\|python_end\|>\s*"
    r"<\|output_start\|>(.*?)<\|output_end\|>", re.S,
)
NUMERIC = re.compile(r"[+-]?" + NUMBER_BODY + r"(?:[eE][+-]?[0-9]+)?")


def score_answer_format(response):
    """One answer marker, on a complete final line; this does not check correctness."""
    lines = response.rstrip().splitlines()
    return float(bool(lines) and response.count("####") == 1
                 and ANSWER_LINE.fullmatch(lines[-1].strip()) is not None)


def _number(text):
    text = text.strip().replace("_", "")
    if len(text) > 96 or NUMERIC.fullmatch(text) is None:
        raise ValueError("Unsupported number")
    if "e" in text.lower() and abs(int(text.lower().split("e")[1])) > 64:
        raise ValueError("Exponent exceeds arithmetic bound")
    value = Fraction(text.replace(",", ""))
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError("Number exceeds arithmetic bound")
    return value


def _same_value(expected, reported):
    if expected.denominator == 1:
        return expected == reported
    # Allow ordinary floating-point calculator rounding, not arbitrary relative error.
    tolerance = Fraction(max(1e-9, 4 * math.ulp(float(expected))))
    return abs(expected - reported) <= tolerance


@lru_cache(maxsize=8192)
def _arithmetic(expression):
    """Return (canonical syntax, exact value) for bounded numeric arithmetic only.

    Syntax is retained: 1+5 is not treated as the reference calculation 3*2.
    Addition/multiplication accept reordered operands. No generated code is executed.
    """
    expression = expression.strip()
    if len(expression) > 256 or re.fullmatch(r"[0-9eE_.+*/()%\s-]+", expression) is None:
        raise ValueError("Expression is too long or contains unsupported syntax")
    root = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(root)) > 64:
        raise ValueError("Expression has too many nodes")

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError("Expression is too deep")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = _number(ast.get_source_segment(expression, node))
            return ("number", value), value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            key, value = visit(node.operand, depth + 1)
            if isinstance(node.op, ast.UAdd):
                return key, value
            return (("number", -value) if key[0] == "number" else ("negative", key)), -value
        if not isinstance(node, ast.BinOp):
            raise ValueError("Only numeric arithmetic is supported")
        left, a = visit(node.left, depth + 1)
        right, b = visit(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            name, value = "add", a + b
        elif isinstance(node.op, ast.Mult):
            name, value = "multiply", a * b
        elif isinstance(node.op, ast.Sub):
            name, value = "subtract", a - b
        elif isinstance(node.op, ast.Div):
            name, value = "divide", a / b
        elif isinstance(node.op, ast.FloorDiv):
            name, value = "floor_divide", Fraction(a // b)
        elif isinstance(node.op, ast.Mod):
            name, value = "modulo", a % b
        elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 8:
            name, value = "power", a ** int(b)
        else:
            raise ValueError("Unsupported arithmetic operator or exponent")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
            raise ValueError("Arithmetic result exceeds bound")
        if name in ("add", "multiply"):
            operands = []
            for key in (left, right):
                operands.extend(key[1] if key[0] == name else (key,))
            key = name, tuple(sorted(operands, key=repr))
        else:
            key = name, left, right
        return key, value

    return visit(root.body)


def _calculation_key(expression, output):
    try:
        key, expected = _arithmetic(expression)
        return key if key[0] != "number" and _same_value(expected, _number(output)) else None
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError):
        return None


@lru_cache(maxsize=4096)
def _reference_keys(calculations, final_answer):
    keys = set()
    try:
        final_value = _number(final_answer)
    except ValueError:
        return frozenset()
    for expression, output in calculations:
        key = _calculation_key(expression, output)
        if key is not None and not _same_value(final_value, _number(output)):
            keys.add(key)
    return frozenset(keys)


def reference_calculation_keys(reference_parts, final_answer):
    """Distinct, arithmetically valid reference calculations before the final value."""
    calculations = tuple((a["text"], b["text"]) for a, b in zip(reference_parts, reference_parts[1:])
                         if a["type"] == "python" and b["type"] == "python_output")
    return _reference_keys(calculations, final_answer)


def score_reference_calculations(response, reference_parts, final_answer):
    """Coverage of distinct, valid reference calculations excluding the final value.

    This conservative proxy cannot recognize every valid alternative solution or
    establish that a calculation is relevant to its surrounding explanation.
    """
    expected = reference_calculation_keys(reference_parts, final_answer)
    if not expected:
        return 0.0
    reasoning = response.split("####", 1)[0]
    found = {_calculation_key(expression, output) for expression, output in TOOL_CALL.findall(reasoning)}
    return len(expected.intersection(found)) / len(expected)


@register("answer_format")
def answer_format(conversation, response, ctx):
    """Partial credit on incorrect responses only; strict-correct responses stay at 1."""
    return 0.0 if ctx["correct"] else score_answer_format(response)


@register("reference_calculation")
def reference_calculation(conversation, response, ctx):
    """Partial reference-calculation credit on incorrect responses only."""
    if ctx["correct"]:
        return 0.0
    return score_reference_calculations(response, ctx["ref_solution_parts"], ctx["ref"])


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
