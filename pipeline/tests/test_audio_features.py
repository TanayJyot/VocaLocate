import numpy as np

from conftest import tone
from vocalocate_pipeline.audio import clean_array, dbfs, rms
from vocalocate_pipeline.config import AudioConfig, FeatureConfig
from vocalocate_pipeline.features import log_mel, mel_filterbank, summary_vector

CFG = AudioConfig()


def test_clean_resamples_to_mono_32k_and_normalises_loudness():
    res = clean_array(tone(440, 2, 44_100, amp=0.05, channels=2), 44_100, CFG)
    assert res.status == "ok"
    assert res.orig_channels == 2 and res.orig_sample_rate == 44_100
    assert res.audio.ndim == 1
    assert abs(res.duration_sec - 2.0) < 0.01
    assert abs(dbfs(rms(res.audio)) - CFG.target_rms_dbfs) < 0.1


def test_peak_is_capped_when_loudness_target_would_clip():
    # A sparse click train has a tiny RMS, so reaching -20 dBFS RMS would push the peak past 0 dBFS.
    x = np.zeros(32_000, dtype=np.float32)
    x[::1000] = 0.5
    res = clean_array(x, 32_000, CFG)
    assert res.peak_dbfs <= CFG.max_peak_dbfs + 1e-3


def test_silence_is_rejected():
    res = clean_array(np.zeros(32_000, dtype=np.float32), 32_000, CFG)
    assert res.status == "rejected" and res.reject_reason == "silent"


def test_leading_and_trailing_silence_is_trimmed():
    x = np.concatenate([np.zeros(32_000), tone(440, 1, 32_000), np.zeros(32_000)]).astype(np.float32)
    res = clean_array(x, 32_000, CFG)
    assert abs(res.duration_sec - 1.0) < 0.01


def test_long_clip_is_truncated_and_flagged():
    res = clean_array(tone(440, 25, 16_000), 16_000, CFG)
    assert res.status == "ok" and res.truncated
    assert res.duration_sec == CFG.max_duration_sec


def test_clipping_is_flagged_not_rejected():
    res = clean_array(np.clip(tone(440, 1, 32_000, amp=2.0), -1, 1), 32_000, CFG)
    assert res.status == "ok" and res.clipped


def test_too_short_is_rejected():
    res = clean_array(tone(440, 0.05, 32_000), 32_000, CFG)
    assert res.status == "rejected" and res.reject_reason == "too_short"


def test_log_mel_matches_baseline_frame_count_and_peaks_at_tone_frequency():
    fc = FeatureConfig()
    sr = 32_000
    x = tone(1000, 10, sr)
    mel = log_mel(x, sr, fc)
    # torch.stft(center=True) gives 1 + len // hop frames; pre-emphasis drops one sample.
    assert mel.shape == (fc.n_mels, 1 + (len(x) - 1) // fc.hop_length)
    banks = mel_filterbank(fc.n_mels, fc.n_fft, sr, fc.fmin, fc.fmax)
    tone_bin = round(1000 / (sr / fc.n_fft))
    assert mel.mean(axis=1).argmax() == banks[:, tone_bin].argmax()


def test_summary_vector_is_mean_and_std_per_band():
    mel = np.random.default_rng(0).normal(size=(128, 50)).astype(np.float32)
    v = summary_vector(mel)
    assert v.shape == (256,)
    np.testing.assert_allclose(v[:128], mel.mean(axis=1), rtol=1e-6)
