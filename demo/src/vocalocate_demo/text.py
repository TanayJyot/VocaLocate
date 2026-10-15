"""The semantic text branch: a frozen CLAP model scores optional text hints against SFX audio.

Stage 6 of the technical flow: optional hints such as "metallic" or "short" are
encoded with CLAP's text encoder, and every SFX with CLAP's audio encoder. CLAP
is used as is, with no fine-tuning. The text score is fused with the voice score
in `fusion.py`.

Backends:
* "hf": LAION CLAP through Hugging Face transformers (default
  `laion/clap-htsat-unfused`). We use this instead of Microsoft's msclap (the
  Elizalde et al. paper cited in A1) because msclap pins numpy<2 and needs a
  CUDA-matched torchaudio, which conflicts with the pipeline's pyarrow. As A2
  says, LAION CLAP's own training list (LAION-Audio-630K: AudioCaps, Clotho,
  Freesound, BBC Sound Effects and more) must be checked before trusting text
  scores on those datasets.
* "fake": a deterministic stand-in for tests: text and audio are hashed into the
  same space, so it runs offline without weights.
"""

from __future__ import annotations

import hashlib
import logging
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

DEFAULT_CLAP = "laion/clap-htsat-unfused"
CLAP_SR = 48_000


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def _load_48k(path: str, max_sec: float = 10.0) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != CLAP_SR:
        g = gcd(sr, CLAP_SR)
        x = resample_poly(x, CLAP_SR // g, sr // g).astype(np.float32)
    return x[: int(max_sec * CLAP_SR)]


class ClapScorer:
    """Embeds text and audio files into CLAP's shared space (unit vectors)."""

    def __init__(self, model_name: str = DEFAULT_CLAP, device: str = "auto", backend: str = "hf", batch_size: int = 32):
        self.model_name, self.backend, self.batch_size = model_name, backend, batch_size
        self.version = f"clap:{model_name}" if backend == "hf" else "clap:fake"
        if backend == "fake":
            return
        if backend != "hf":
            raise ValueError(f"unknown CLAP backend {backend!r}")
        import torch
        from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, torch.device(device)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.features = ClapFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -- text ---------------------------------------------------------------
    def embed_text(self, texts: list[str]) -> np.ndarray:
        if self.backend == "fake":
            return _normalize(np.stack([_fake_text(t) for t in texts])) if texts else np.zeros((0, 64), np.float32)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                tok = self.tokenizer(texts[i : i + self.batch_size], padding=True, return_tensors="pt").to(self.device)
                out.append(self.model.get_text_features(**tok).float().cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)

    # -- audio --------------------------------------------------------------
    def embed_audio(self, paths: list[str]) -> np.ndarray:
        if self.backend == "fake":
            return _normalize(np.stack([_fake_audio(p) for p in paths])) if paths else np.zeros((0, 64), np.float32)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                waves = [_load_48k(p) for p in paths[i : i + self.batch_size]]
                # Fused CLAP models take 4-channel "fusion" input; unfused ones take one channel. Clips are cut
                # to 10 s above, so rand_trunc never actually truncates and embeddings stay deterministic.
                fusion = getattr(self.model.config.audio_config, "enable_fusion", False)
                feats = self.features(
                    waves, sampling_rate=CLAP_SR, return_tensors="pt", truncation="fusion" if fusion else "rand_trunc"
                )
                feats = {k: v.to(self.device) for k, v in feats.items() if k in ("input_features", "is_longer")}
                out.append(self.model.get_audio_features(**feats).float().cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)


# Fake backend: words and file names hash to fixed random directions, so a hint
# like "metallic" lands near SFX whose metadata-derived filename says "metal".
_FAKE_DIM = 64


def _word_vec(word: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(word.lower().encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(_FAKE_DIM)


def _fake_text(text: str) -> np.ndarray:
    words = [w for w in text.replace("-", " ").replace("_", " ").split() if w] or ["empty"]
    return np.sum([_word_vec(w) for w in words], axis=0)


def _fake_audio(path: str) -> np.ndarray:
    x, _ = sf.read(path, dtype="float32", always_2d=True)
    seed = int.from_bytes(hashlib.sha256(x.tobytes()[:65536]).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(_FAKE_DIM)


def cached_audio_embeddings(cfg, part: str, clip_ids: list[str], paths: list[str], clap: ClapScorer) -> np.ndarray:
    """CLAP audio embeddings for clips, cached in the lake's gold `embeddings` table (tower "clap_audio").

    Only clips without a cached vector for this CLAP version are embedded; the part
    is rewritten with exactly `clip_ids`, so clips that left a library drop out.
    """
    from vocalocate_pipeline.lake import Lake
    from vocalocate_pipeline.schemas import to_table

    lake = Lake(cfg.lake_root)
    cached = lake.read("embeddings", part)
    known = {
        r["clip_id"]: r["vector"]
        for r in (cached.to_pylist() if cached is not None else [])
        if r["model_version"] == clap.version
    }
    missing = [(cid, p) for cid, p in zip(clip_ids, paths) if cid not in known]
    if missing:
        log.info("CLAP: embedding %d clips (%d cached)", len(missing), len(clip_ids) - len(missing))
        for (cid, _), vec in zip(missing, clap.embed_audio([p for _, p in missing])):
            known[cid] = vec.tolist()
    rows = [
        {"clip_id": cid, "source": part.split("__")[0], "model_version": clap.version, "tower": "clap_audio",
         "vector": known[cid]}
        for cid in dict.fromkeys(clip_ids)
    ]
    lake.write("embeddings", part, to_table("embeddings", rows))
    return np.array([known[cid] for cid in clip_ids], dtype=np.float32).reshape(len(clip_ids), -1)
