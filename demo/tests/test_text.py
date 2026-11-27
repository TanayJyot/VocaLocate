"""Test CLAP audio preprocessing and model input/output contracts offline."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch

from vocalocate_demo.text import CLAP_SR, ClapScorer, _load_48k


def test_load_clap_audio_resamples_stereo_and_limits_duration(tmp_path):
    path = tmp_path / "stereo.wav"
    sr = CLAP_SR // 2
    stereo = np.tile([0.25, 0.5], (sr // 10, 1))
    sf.write(path, stereo, sr, subtype="FLOAT")

    audio = _load_48k(str(path), max_sec=0.05)

    assert audio.shape == (2_400,)
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio[20:-20], 0.375, atol=1e-3)


class TokenBatch(dict):
    def to(self, device):
        return {name: tensor.to(device) for name, tensor in self.items()}


@pytest.mark.parametrize("enable_fusion", [False, True])
def test_clap_batches_and_normalizes_both_feature_return_types(tmp_path, monkeypatch, enable_fusion):
    device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.config.audio_config.enable_fusion = enable_fusion
    model.get_text_features.side_effect = lambda input_ids: 3.0 * input_ids
    model.get_audio_features.side_effect = lambda input_features, is_longer: SimpleNamespace(
        pooler_output=4.0 * input_features,
    )
    tokenizer = Mock(side_effect=lambda prompts, **kwargs: TokenBatch(
        input_ids=torch.tensor([[1.0, 0.0] if p == "wind" else [0.0, 1.0] for p in prompts]),
    ))

    def featurize(waves, **kwargs):
        return {
            "input_features": torch.tensor([
                [1.0, 0.0] if wave[0] > 0 else [0.0, 1.0] for wave in waves
            ]),
            "is_longer": torch.zeros((len(waves), 1), dtype=torch.bool),
            "ignored_metadata": object(),
        }

    extractor = Mock(side_effect=featurize)
    transformers = SimpleNamespace(
        AutoTokenizer=Mock(), ClapFeatureExtractor=Mock(), ClapModel=Mock(),
    )
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.ClapFeatureExtractor.from_pretrained.return_value = extractor
    transformers.ClapModel.from_pretrained.return_value = model
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    paths = []
    for index, sign in enumerate([1, -1, 1]):
        path = tmp_path / f"clip-{index}.wav"
        sf.write(path, np.full(100, sign * 0.25), CLAP_SR)
        paths.append(str(path))

    scorer = ClapScorer(model_name="fixture-clap", batch_size=2)
    expected = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    np.testing.assert_allclose(scorer.embed_text(["wind", "clang", "wind"]), expected)
    np.testing.assert_allclose(scorer.embed_audio(paths), expected)
    assert scorer.embed_text([]).shape == scorer.embed_audio([]).shape == (0, 512)
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    transformers.ClapModel.from_pretrained.assert_called_once_with("fixture-clap")
    assert [c.args[0] for c in tokenizer.call_args_list] == [["wind", "clang"], ["wind"]]
    assert [len(c.args[0]) for c in extractor.call_args_list] == [2, 1]
    assert all(c.kwargs == {
        "sampling_rate": CLAP_SR,
        "return_tensors": "pt",
        "truncation": "fusion" if enable_fusion else "rand_trunc",
    } for c in extractor.call_args_list)


def test_clap_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown CLAP backend"):
        ClapScorer(backend="invalid")
