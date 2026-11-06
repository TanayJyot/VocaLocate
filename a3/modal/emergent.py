"""
A3 Part 5: questions the d20 nanochat can answer and the d12 picochat cannot. Every model gets the same fixed prompts,
with greedy decoding, and an answer counts only if an accepted keyword appears on the first line it generates.
The prompt list and keywords were fixed before any model was run, so the hit rate can be reported along with the picks.

    modal run emergent.py
"""
import json
import os
import re
import sys

import modal

NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests", "jinja2", "datasets",
                 "kernels==0.11.7", "huggingface-hub==0.34.4", "rustbpe", "wandb")
    .add_local_dir(NANOCHAT, "/root/nanochat", ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"])
)
vol = modal.Volume.from_name("vocalocate-a3")
app = modal.App("vocalocate-a3-emergent", image=image)
BASE = "/vol/a3"

MODELS = ["d12-baseline", "d12-looped-l2-embed", "d20-combined"]
MAX_TOKENS = 30
# (question, accepted answers as regexes)
QUESTIONS = [
    # Geography
    ("What is the capital of Australia?", [r"canberra"]),
    ("What is the capital of Canada?", [r"ottawa"]),
    ("What is the capital of Japan?", [r"tokyo"]),
    ("What is the capital of Egypt?", [r"cairo"]),
    ("What is the capital of Brazil?", [r"bras[ií]lia"]),
    ("What is the longest river in Africa?", [r"\bnile\b"]),
    ("What is the largest ocean on Earth?", [r"pacific"]),
    ("What is the tallest mountain in the world?", [r"everest"]),
    ("On which continent is the Sahara Desert?", [r"africa"]),
    ("Which country has the largest population in South America?", [r"brazil"]),
    # Science
    ("What is the chemical symbol for gold?", [r"\bau\b"]),
    ("What is the chemical symbol for sodium?", [r"\bna\b"]),
    ("What gas do plants absorb from the air?", [r"carbon dioxide", r"\bco2\b"]),
    ("What is the chemical formula for water?", [r"h2o", r"h₂o"]),
    ("Which planet is known as the Red Planet?", [r"\bmars\b"]),
    ("What is the largest planet in the solar system?", [r"jupiter"]),
    ("What force keeps the planets in orbit around the Sun?", [r"gravity", r"gravitational"]),
    ("What organ pumps blood through the human body?", [r"\bheart\b"]),
    ("What part of the cell contains DNA?", [r"nucleus"]),
    ("What is the boiling point of water in degrees Celsius?", [r"\b100\b"]),
    ("How many bones are in the adult human body?", [r"\b206\b"]),
    ("Who developed the theory of general relativity?", [r"einstein"]),
    ("Who proposed the theory of evolution by natural selection?", [r"darwin"]),
    # History
    ("Who was the first president of the United States?", [r"washington"]),
    ("In what year did World War II end?", [r"\b1945\b"]),
    ("In what year did World War I begin?", [r"\b1914\b"]),
    ("In what year did humans first land on the Moon?", [r"\b1969\b"]),
    ("Who was the first person to walk on the Moon?", [r"armstrong"]),
    ("Which empire built the Colosseum?", [r"\broman\b", r"\brome\b"]),
    ("Who wrote the Declaration of Independence?", [r"jefferson"]),
    ("Who was the British prime minister for most of World War II?", [r"churchill"]),
    # Literature and arts
    ("Who wrote Romeo and Juliet?", [r"shakespeare"]),
    ("Who wrote Pride and Prejudice?", [r"austen"]),
    ("Who wrote the novel 1984?", [r"orwell"]),
    ("Who painted the Mona Lisa?", [r"leonardo", r"da vinci"]),
    ("Who painted The Starry Night?", [r"van gogh"]),
    ("Who composed the Ninth Symphony known for the Ode to Joy?", [r"beethoven"]),
    ("Who wrote The Great Gatsby?", [r"fitzgerald"]),
    # Technology and general knowledge
    ("Who founded Microsoft with Paul Allen?", [r"gates"]),
    ("Who invented the telephone?", [r"\bbell\b"]),
    ("What does DNA stand for?", [r"deoxyribonucleic"]),
    ("What does CPU stand for?", [r"central processing unit"]),
    ("What currency is used in Japan?", [r"\byen\b"]),
    ("What language is mainly spoken in Brazil?", [r"portuguese"]),
    ("What is the square root of 144?", [r"\b12\b"]),
    ("What is 7 times 8?", [r"\b56\b"]),
    ("How many continents are there?", [r"\bseven\b", r"\b7\b"]),
    ("How many days are there in a leap year?", [r"\b366\b"]),
    ("What is the freezing point of water in degrees Fahrenheit?", [r"\b32\b"]),
    ("What is the opposite of hot?", [r"\bcold\b"]),
]


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=1800)
def ask(tags, questions):
    os.environ.update(NANOCHAT_BASE_DIR=BASE, OMP_NUM_THREADS="1")
    sys.path.insert(0, "/root/nanochat")
    import torch
    from nanochat.checkpoint_manager import load_model

    vol.reload()
    answers = {}
    for tag in tags:
        model, tokenizer, _ = load_model("base", torch.device("cuda"), phase="eval", model_tag=tag)
        answers[tag] = []
        for q in questions:
            tokens = tokenizer(f"Question: {q}\nAnswer:", prepend="<|bos|>")
            out = []
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for token in model.generate(tokens, max_tokens=MAX_TOKENS, temperature=0):
                    out.append(token)
            answers[tag].append(tokenizer.decode(out))
        del model
        torch.cuda.empty_cache()
    return answers


def graded(text, accepted):
    first = text.strip().split("\n")[0].lower()
    return any(re.search(a, first) for a in accepted)


@app.local_entrypoint()
def run_emergent():
    answers = ask.remote(MODELS, [q for q, _ in QUESTIONS])
    rows = []
    for i, (q, accepted) in enumerate(QUESTIONS):
        rows.append({"question": q, **{t: answers[t][i].strip().split("\n")[0] for t in MODELS},
                     **{f"{t}_ok": graded(answers[t][i], accepted) for t in MODELS}})
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergent_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    for t in MODELS:
        print(t, sum(r[f"{t}_ok"] for r in rows), "of", len(rows))
    print("d20 right, d12 looped-l2 wrong:", sum(r["d20-combined_ok"] and not r["d12-looped-l2-embed_ok"] for r in rows))
    print("d12 looped-l2 right, d20 wrong:", sum(r["d12-looped-l2-embed_ok"] and not r["d20-combined_ok"] for r in rows))
