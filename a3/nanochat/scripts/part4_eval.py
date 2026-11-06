"""Compare two Part 4 checkpoints on the same length- and distance-controlled passkey task."""
import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import random

import torch

from nanochat.checkpoint_manager import build_model
from nanochat.tokenizer import get_tokenizer

CONDITIONS = ((512, 128), (512, 384), (1024, 128), (1024, 384), (1024, 768),
              (2048, 128), (2048, 384), (2048, 768), (2048, 1536))
CODE_WORDS = ("amber", "cobalt", "copper", "silver", "walnut", "maple", "cedar", "granite",
              "quartz", "olive", "coral", "violet", "ivory", "honey", "satin", "silk")
FILLER = (
    " The library opens in the morning and closes after the final evening bus.",
    " A group of students met beside the station to discuss plans for the weekend.",
    " The first report explained how the neighborhood changed over several decades.",
    " At the market, vendors carried boxes from a truck and arranged them on tables.",
    " The committee delayed the vote until everyone could read the revised proposal.",
    " The gardener replaced a broken fence and moved the plants away from the path.",
    " Rain began during the afternoon and continued until the streets were empty.",
    " A trail follows the edge of the hill before turning down toward the bridge.",
    " The local theater plans to present a different play at the end of each month.",
    " Workers painted the walls, repaired the windows and installed new flooring.",
    " After dinner the family walked through the village and returned before dark.",
    " A teacher asked the class to compare the results from two earlier experiments.",
)


@dataclass(frozen=True)
class PasskeyCase:
    tokens: list[int]
    candidates: list[int]
    correct: int


class PasskeyTask:
    def __init__(self, tokenizer):
        self.bos = tokenizer.get_bos_token_id()
        self.needle = tokenizer.encode("\nFor locker 42, the access code is")
        self.ending = tokenizer.encode(".\n")
        self.query = tokenizer.encode("\nAs written above, the access code for locker 42 is")
        self.filler = [tokenizer.encode(sentence) for sentence in FILLER]
        other_tokens = set(self.needle + self.ending + self.query).union(*self.filler)
        encodings = [tokenizer.encode(" " + word) for word in CODE_WORDS]
        self.codes = [ids[0] for ids in encodings if len(ids) == 1 and ids[0] not in other_tokens]
        if len(self.codes) < 4:
            raise ValueError("The tokenizer needs at least four distinct single-token codes")

    def case(self, length, distance, seed):
        rng = random.Random(seed)
        candidates = rng.sample(self.codes, 4)
        correct = rng.randrange(len(candidates))
        after = distance - len(self.ending) - len(self.query)
        before = length - 2 - len(self.needle) - distance
        if min(before, after) < 0:
            raise ValueError("The requested length and distance leave no room for the prompt")
        filler = []
        while len(filler) < before + after:
            filler.extend(rng.choice(self.filler))
        tokens = ([self.bos] + filler[:before] + self.needle + [candidates[correct]] +
                  self.ending + filler[before:before + after] + self.query)
        return PasskeyCase(tokens, candidates, correct)


class CompetingPasskeyTask(PasskeyTask):
    LOCKERS = (17, 42, 58, 93)

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.record_prefixes = {
            key: tokenizer.encode(f"\nFor locker {key}, the access code is") for key in self.LOCKERS
        }
        self.queries = {
            key: tokenizer.encode(f"\nAs written above, the access code for locker {key} is")
            for key in self.LOCKERS
        }
        assert {len(tokens) for tokens in self.record_prefixes.values()} == {len(self.needle)}
        assert {len(tokens) for tokens in self.queries.values()} == {len(self.query)}

    def case(self, length, distance, seed):
        original = super().case(length, distance, seed)
        rng = random.Random(seed + 381372)
        target_key = rng.choice(self.LOCKERS)
        other_keys = [key for key in self.LOCKERS if key != target_key]
        other_codes = [code for index, code in enumerate(original.candidates) if index != original.correct]
        rng.shuffle(other_codes)

        # Randomize the target record's order while keeping its distance fixed.
        earlier_count = rng.randrange(4)
        earlier = rng.sample(range(distance + 32, length - 24, 32), earlier_count)
        later = rng.sample(range(32, distance - 23, 32), 3 - earlier_count)
        other_distances = earlier + later
        rng.shuffle(other_distances)

        tokens = original.tokens.copy()
        target_position = length - 1 - distance
        tokens[target_position - len(self.needle):target_position] = self.record_prefixes[target_key]
        tokens[-len(self.query):] = self.queries[target_key]
        for key, code, other_distance in zip(other_keys, other_codes, other_distances):
            position = length - 1 - other_distance
            start = position - len(self.needle)
            stop = position + 1 + len(self.ending)
            assert start > 0 and stop <= length - len(self.query)
            assert not any(token in original.candidates for token in tokens[start:stop])
            tokens[start:stop] = self.record_prefixes[key] + [code] + self.ending
        assert len(tokens) == length
        assert tokens[target_position] == original.candidates[original.correct]
        assert all(tokens.count(code) == 1 for code in original.candidates)
        return replace(original, tokens=tokens)


@torch.inference_mode()
def score(model, cases, device, batch_size, *, predictions=None):
    correct = 0
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        ids = torch.tensor([case.tokens for case in batch], device=device)
        candidates = torch.tensor([case.candidates for case in batch], device=device)
        answers = torch.tensor([case.correct for case in batch], device=device)
        logits = model(ids)[:, -1, :].gather(1, candidates)
        predicted = logits.argmax(dim=1)
        correct += (predicted == answers).sum().item()
        if predictions is not None:
            predictions.extend(predicted.tolist())
    return correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs=2, default=(1260, 2520))
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--competing-records", action="store_true",
                        help="use four locker records per prompt and save per-trial predictions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.batch_size) < 1:
        parser.error("--trials and --batch-size must be positive")

    os.environ.setdefault("NANOCHAT_BASE_DIR", str(Path(__file__).resolve().parents[2]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_class = CompetingPasskeyTask if args.competing_records else PasskeyTask
    task = task_class(get_tokenizer())
    results = []
    for step in args.steps:
        model, _, meta = build_model(args.checkpoint_dir, step, device, "eval", eval_sequence_len=2048)
        trained_length = meta["model_config"]["sequence_len"]
        for length, distance in CONDITIONS:
            cases = [task.case(length, distance, args.seed + i) for i in range(args.trials)]
            predictions = [] if args.competing_records else None
            correct = score(model, cases, device, args.batch_size, predictions=predictions)
            row = dict(step=step, trained_length=trained_length, input_tokens=length,
                       distance_tokens=distance, extrapolation=length > trained_length,
                       correct=correct, trials=args.trials, accuracy=correct / args.trials)
            print(json.dumps(row), flush=True)
            if predictions is not None:
                row["case_results"] = [
                    dict(seed=args.seed + index, candidate_token_ids=case.candidates,
                         target_index=case.correct, predicted_index=prediction)
                    for index, (case, prediction) in enumerate(zip(cases, predictions, strict=True))
                ]
            results.append(row)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(seed=args.seed, device=str(device), chance=0.25,
                                         results=results), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
