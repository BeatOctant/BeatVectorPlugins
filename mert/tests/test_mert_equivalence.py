"""octantvector-mert against the OctantVector 0.3 code (encoders/mert.py, vocal.py; copied below as the reference), both
loading the same tiny randomly initialised MERT from a temporary models_dir. No network, no real weights."""
import json

import numpy as np
import pytest

import octantvector_mert as plug
from octantvector import plugin

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A models_dir holding a tiny random MERT (10 layers) under the pinned path, with the real config's auto_map (the
    model folder never holds the .py files, so following it would fail), and empty HTDemucs files (the separator is
    replaced). OctantVector reads models_dir from BV_MODELS_DIR outside contract 1."""
    from transformers import Wav2Vec2FeatureExtractor

    from octantvector_mert.configuration_MERT import MERTConfig
    from octantvector_mert.modeling_MERT import MERTModel
    torch.manual_seed(0)
    d = tmp_path / "models" / plug.MERT.path
    d.mkdir(parents=True)
    cfg = MERTConfig(hidden_size=32, num_hidden_layers=10, num_attention_heads=2, intermediate_size=64,
                     conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    save_bin(MERTModel(cfg).eval(), d)
    c = json.loads((d / "config.json").read_text())
    c["auto_map"] = {"AutoConfig": "configuration_MERT.MERTConfig", "AutoModel": "modeling_MERT.MERTModel"}
    (d / "config.json").write_text(json.dumps(c))
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=24000, padding_value=0.0, do_normalize=True,
                             return_attention_mask=True).save_pretrained(d)
    assert sorted(p.name for p in d.iterdir()) == sorted(plug.MERT.files)
    dem = tmp_path / "models" / "hf/adefossez/HTDemucs/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"
    dem.mkdir(parents=True)
    for f in ("htdemucs.yaml", "955717e8.safetensors"):
        (dem / f).write_text("")
    monkeypatch.setenv("BV_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("BV_MODELS_DIR", str(tmp_path / "models"))
    return d


def save_bin(model, d):
    """The model as the pinned repo has it: config.json and pytorch_model.bin (no safetensors)."""
    model.save_pretrained(d)
    (d / "model.safetensors").unlink()
    torch.save(model.state_dict(), d / "pytorch_model.bin")


def songs(n, sr=22050, seconds=40, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(sr * seconds) / sr
    return [(0.3 * np.sin(2 * np.pi * (110 + 50 * i) * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)
            for i in range(n)]


class OldMert:
    """OctantVector 0.3 encoders/mert.py _Mert/Mert95, loading the copied model class from the local folder (the old code
    loaded the same class from the hub with trust_remote_code)."""
    sr, clip_s, n_clips, layers = 24000, 10, 3, (4, 5, 6, 7, 8)

    def __init__(self, folder):
        from transformers import Wav2Vec2FeatureExtractor

        from octantvector_mert.modeling_MERT import MERTModel
        self.proc = Wav2Vec2FeatureExtractor.from_pretrained(folder)
        self.model = MERTModel.from_pretrained(folder).eval()
        self.model.encoder.layers = self.model.encoder.layers[:max(self.layers)]  # keep_layers
        self._cap = {}
        for i in self.layers:
            self.model.encoder.layers[i - 1].register_forward_hook(
                lambda mod, inp, out, i=i: self._cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))

    def prep(self, y, sr):
        cl = [plugin.resample(c, sr, self.sr) for c in plugin.clips(y, sr, self.clip_s, self.n_clips)]
        return self.proc(cl, sampling_rate=self.sr, return_tensors="np")["input_values"].astype(np.float32)

    def clip_vectors(self, x):
        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            self._cap.clear()
            self.model(torch.from_numpy(x).to("cpu", dtype))
            return torch.stack([self._cap[i].float() for i in self.layers]).mean(2).mean(0).cpu().numpy()

    def embed(self, x):
        return self.clip_vectors(x).mean(0)

    def embed_clips(self, voc):  # vocal.py VocalMert95.embed_clips
        x = self.proc([plugin.resample(v, 44100, self.sr) for v in voc], sampling_rate=self.sr,
                      return_tensors="np")["input_values"].astype(np.float32)
        return self.clip_vectors(x)


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


def test_mert_matches_the_old_code(models_dir):
    old, new = OldMert(models_dir), plug.Mert95()
    new.setup_worker()
    new.setup_model("cpu")
    assert len(new.model.encoder.layers) == 8  # the layers after the last one taken are dropped
    for y in songs(2) + [songs(1, seconds=5)[0]]:  # a short song is padded to one clip
        a, b = old.embed(old.prep(y, 22050)), new.embed(new.prep(y, 22050))
        assert a.shape == b.shape == (32,)
        assert np.abs(a - b).max() < 1e-5


def test_vocal_mert_matches_the_old_code(models_dir, monkeypatch):
    from octantvector.generation.encoders import stem  # the separator has no plugin hook: replaced here
    monkeypatch.setattr(stem, "Separator", FakeSeparator)
    old, new = OldMert(models_dir), plug.VocalMert95()
    new.setup_worker()
    new.setup_model("cpu")
    ys = [y * g for y, g in zip(songs(3), (2.0, 0.5, 2.0))]  # the quiet one is unvoiced
    prepped = [new.prep(y, 22050) for y in ys]
    for p, y in zip(prepped, ys):
        assert np.array_equal(p, old_vocal_prep(y, 22050))
    want, got = old_vocal_embed_many(old, prepped), new.embed_many(prepped)
    assert [w is None for w in want] == [g is None for g in got] == [False, True, False]
    for w, g in zip(want, got):
        if w is not None:
            assert np.abs(w - g).max() < 1e-5


def test_a_missing_model_file_names_the_model(models_dir):
    (models_dir / "pytorch_model.bin").unlink()
    with pytest.raises(plugin.GenerationError, match=f"m-a-p/MERT-v1-95M@{plug.MERT.revision} is not in"):
        plug.Mert95().setup_model("cpu")
