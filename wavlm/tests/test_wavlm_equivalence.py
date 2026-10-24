"""octantvector-wavlm against the OctantVector 0.3 code (encoders/vocal.py VocalWavlm and VocalStem; copied below as the
reference), both loading the same tiny randomly initialised WavLM x-vector model from a temporary models_dir, with the
separator replaced by a deterministic fake. No network, no real weights."""
import numpy as np
import pytest

import octantvector_wavlm as plug
from octantvector import plugin

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A models_dir holding a tiny random WavLM x-vector model under the pinned path, and empty HTDemucs files (the
    separator is replaced). OctantVector reads models_dir from BV_MODELS_DIR outside contract 1."""
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMForXVector
    torch.manual_seed(0)
    d = tmp_path / "models" / plug.WAVLM.path
    d.mkdir(parents=True)
    cfg = WavLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
                      conv_dim=(8,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                      tdnn_dim=(8, 8, 8, 8, 24), xvector_output_dim=8)
    model = WavLMForXVector(cfg).eval()
    model.save_pretrained(d)  # as the pinned repo has it: config.json and pytorch_model.bin
    (d / "model.safetensors").unlink()
    torch.save(model.state_dict(), d / "pytorch_model.bin")
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=True).save_pretrained(d)
    assert sorted(p.name for p in d.iterdir()) == sorted(plug.WAVLM.files)
    dem = tmp_path / "models" / "hf/adefossez/HTDemucs/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"
    dem.mkdir(parents=True)
    for f in ("htdemucs.yaml", "955717e8.safetensors"):
        (dem / f).write_text("")
    monkeypatch.setenv("BV_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("BV_MODELS_DIR", str(tmp_path / "models"))
    return d


def songs(n, sr=22050, seconds=40, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(sr * seconds) / sr
    return [(0.3 * np.sin(2 * np.pi * (110 + 50 * i) * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)
            for i in range(n)]


class OldWavlm:
    """OctantVector 0.3 encoders/vocal.py VocalWavlm, loading from the local folder instead of the hub."""
    sr = 16000

    def __init__(self, folder):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.proc = AutoFeatureExtractor.from_pretrained(folder)
        self.model = WavLMForXVector.from_pretrained(folder).eval()

    def embed_clips(self, voc):
        f = self.proc([plugin.resample(v, 44100, self.sr) for v in voc], sampling_rate=self.sr, return_tensors="pt",
                      padding=True)
        with torch.inference_mode():
            e = self.model(**f).embeddings
            return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def fake_vocals(mix):
    """Deterministic stand-in for the separator: loud songs keep a shaped 'voice', quiet ones almost none."""
    loud = np.abs(mix).max(1, keepdims=True) > 0.5
    env = 0.5 + 0.5 * np.sin(np.linspace(0, 20, mix.shape[1], dtype=np.float32))
    return np.where(loud, 0.7 * mix * env, 0.001 * mix).astype(np.float32)


class FakeSeparator:
    def __init__(self, folder, device):
        pass

    def vocals(self, mix):
        return fake_vocals(mix)


def old_vocal_embed_many(old, prepped, min_vocal_share=0.3):
    """OctantVector 0.3 encoders/vocal.py VocalStem.embed_many with its vocal_share."""
    def vocal_share(voc, mix, sr, frame=0.1, ratio=0.1):
        n = int(frame * sr)
        k = voc.shape[-1] // n
        ev = (voc[..., :k * n].reshape(len(voc), k, n) ** 2).mean(-1)
        em = (mix[..., :k * n].reshape(len(mix), k, n) ** 2).mean(-1) + 1e-10
        return ((ev / em) >= ratio).mean(-1)
    mix = np.concatenate(prepped)
    voc = fake_vocals(mix)
    bounds = np.cumsum([0] + [len(p) for p in prepped])
    voiced = [float(vocal_share(voc[a:b], mix[a:b], 44100).mean()) >= min_vocal_share
              for a, b in zip(bounds[:-1], bounds[1:])]
    keep = [j for c, a, b in zip(voiced, bounds[:-1], bounds[1:]) if c for j in range(a, b)]
    if not keep:
        return [None] * len(prepped)
    vecs = dict(zip(keep, old.embed_clips(voc[keep])))
    return [np.mean([vecs[i] for i in range(a, b)], 0) if c else None
            for c, a, b in zip(voiced, bounds[:-1], bounds[1:])]


def old_vocal_prep(y, sr, clip_s=7.8, n_clips=3):
    n = int(clip_s * 44100)
    cl = [plugin.resample(c, sr, 44100)[:n] for c in plugin.clips(y, sr, clip_s, n_clips)]
    return np.stack([np.pad(c, (0, n - len(c))) for c in cl]).astype(np.float32)


@pytest.fixture
def fake_separator(monkeypatch):
    from octantvector.generation.encoders import stem  # the separator has no plugin hook: replaced here
    monkeypatch.setattr(stem, "Separator", FakeSeparator)


def test_vocal_wavlm_matches_the_old_code(models_dir, fake_separator):
    old, new = OldWavlm(models_dir), plug.VocalWavlm()
    new.setup_worker()
    new.setup_model("cpu")
    ys = [y * g for y, g in zip(songs(3), (2.0, 0.5, 2.0))]  # the quiet one is unvoiced
    ys.append(songs(1, seconds=5, seed=2)[0] * 2.0)  # a short song: one padded clip
    prepped = [new.prep(y, 22050) for y in ys]
    for p, y in zip(prepped, ys):
        assert np.array_equal(p, old_vocal_prep(y, 22050))
    want, got = old_vocal_embed_many(old, prepped), new.embed_many(prepped)
    assert [w is None for w in want] == [g is None for g in got] == [False, True, False, False]
    for w, g in zip(want, got):
        if w is not None:
            assert g.shape == (8,) and np.abs(w - g).max() < 1e-5


def test_all_unvoiced_gives_no_vectors(models_dir, fake_separator):
    new = plug.VocalWavlm()
    new.setup_model("cpu")
    assert new.embed(new.prep(songs(1)[0] * 0.5, 22050)) is None
