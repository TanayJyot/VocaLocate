"""Check the MobileNet retrieval wrapper with local, randomly initialized weights."""

import pytest
import torch

from vocalocate_demo.model import DualEncoder


@pytest.mark.parametrize("shared", [False, True])
def test_mobilenet_towers_start_with_equal_unit_embeddings(shared):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(490)
        model = DualEncoder(encoder="mobilenetv3", pretrained=None, frames=64, shared=shared).eval()
        log_mel = torch.randn(2, 128, 64)

    assert (model.imitation is model.sfx) is shared
    assert all(not parameter.requires_grad for parameter in model.imitation.net.classifier.parameters())
    assert any(parameter.requires_grad for parameter in model.imitation.parameters())
    with torch.inference_mode():
        voice = model.embed_imitation(log_mel)
        sfx = model.embed_sfx(log_mel)
    assert voice.shape == (2, 960)
    torch.testing.assert_close(voice.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(voice, sfx)


def test_dual_encoder_reports_invalid_architecture():
    with pytest.raises(ValueError, match="unknown encoder"):
        DualEncoder(encoder="unsupported")
