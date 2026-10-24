"""octantvector-clap against the OctantVector 0.3 code (encoders/clap.py, clap_concepts.py, vocal.py; copied below as the
reference), both loading the same tiny randomly initialised CLAP from a temporary models_dir. No network, no real
weights."""
import json

import numpy as np
import pytest

import octantvector_clap as plug
from octantvector import plugin

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A models_dir holding a tiny random CLAP under the pinned path (and empty HTDemucs files: the separator is
    replaced). OctantVector reads models_dir from BV_MODELS_DIR outside contract 1."""
    from transformers import (ClapAudioConfig, ClapConfig, ClapFeatureExtractor, ClapModel, ClapTextConfig,
                              RobertaTokenizerFast)
    torch.manual_seed(0)
    d = tmp_path / "models" / plug.CLAP.path
    d.mkdir(parents=True)
    words = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] + sorted({w for c in plug.CONCEPTS for w in c.split()})
    (tmp_path / "vocab.json").write_text(json.dumps({w: i for i, w in enumerate(words)}))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tok = RobertaTokenizerFast(vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt"),
                               add_prefix_space=False)
    cfg = ClapConfig(text_config=ClapTextConfig(vocab_size=len(words) + 4, hidden_size=16, num_hidden_layers=1,
                                                num_attention_heads=1, intermediate_size=32,
                                                max_position_embeddings=64, projection_dim=16),
                     audio_config=ClapAudioConfig(depths=[1, 1, 1, 1], num_attention_heads=[1, 1, 1, 1],
                                                  patch_embeds_hidden_size=8, hidden_size=64, projection_dim=16),
                     projection_dim=16)
    save_bin(ClapModel(cfg).eval(), d)
    # laid out as the pinned repo is (no processor_config.json): the extractor, the tokenizer and its source files
    ClapFeatureExtractor(truncation="rand_trunc").save_pretrained(d)
    tok.save_pretrained(d)
    for f in ("vocab.json", "merges.txt"):
        (d / f).write_bytes((tmp_path / f).read_bytes())
    if not (d / "special_tokens_map.json").exists():  # newer transformers keep these in tokenizer_config.json
        (d / "special_tokens_map.json").write_text(json.dumps({k: str(v) for k, v in tok.special_tokens_map.items()}))
    assert sorted(p.name for p in d.iterdir()) == sorted(plug.CLAP.files)
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


class OldClap:
    """OctantVector 0.3 encoders/clap.py Clap, loading from the local folder instead of the hub."""
    sr, clip_s, n_clips = 48000, 10, 3

    def __init__(self, folder):
        from transformers import ClapModel, ClapProcessor
        self.proc = ClapProcessor.from_pretrained(folder)
        self.model = ClapModel.from_pretrained(folder).eval()
        self.folder = folder

    def prep(self, y, sr):
        cl = [plugin.resample(c, sr, self.sr) for c in plugin.clips(y, sr, self.clip_s, self.n_clips)]
        f = self.proc(audio=cl, sampling_rate=self.sr, return_tensors="np")
        return f["input_features"].astype(np.float32), f["is_longer"]

    def embed(self, prepped):
        feats, longer = prepped
        with torch.no_grad():
            a = self.model.get_audio_features(input_features=torch.from_numpy(feats), is_longer=torch.from_numpy(longer))
            a = getattr(a, "pooler_output", a)
            return torch.nn.functional.normalize(a, dim=-1).mean(0).cpu().numpy()

    def embed_clips(self, voc):  # vocal.py VocalClap.embed_clips
        f = self.proc(audio=[plugin.resample(v, 44100, self.sr) for v in voc], sampling_rate=self.sr,
                      return_tensors="pt")
        with torch.inference_mode():
            a = self.model.get_audio_features(input_features=f["input_features"], is_longer=f["is_longer"])
            a = getattr(a, "pooler_output", a)
            return torch.nn.functional.normalize(a, dim=-1).cpu().numpy()

    def text_embeddings(self, texts):
        with torch.no_grad():
            t = self.model.get_text_features(**self.proc(text=list(texts), return_tensors="pt", padding=True))
            t = getattr(t, "pooler_output", t)
        return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()


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


def test_clap_matches_the_old_code(models_dir):
    old, new = OldClap(models_dir), plug.Clap()
    new.setup_worker()
    new.setup_model("cpu")
    for y in songs(2) + [songs(1, seconds=5)[0]]:  # a short song is padded to one clip
        a, b = old.embed(old.prep(y, 22050)), new.embed(new.prep(y, 22050))
        assert a.shape == b.shape == (16,)
        assert np.abs(a - b).max() < 1e-5


def test_vocal_clap_matches_the_old_code(models_dir, monkeypatch):
    from octantvector.generation.encoders import stem  # the separator has no plugin hook: replaced here
    monkeypatch.setattr(stem, "Separator", FakeSeparator)
    old, new = OldClap(models_dir), plug.VocalClap()
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


def test_clap_concepts_match_the_old_code(models_dir):
    old, new = OldClap(models_dir), plug.ClapConcepts()
    A = np.random.default_rng(3).standard_normal((5, 16)).astype(np.float32)
    T = old.text_embeddings(plug.CONCEPTS)
    want = (A / np.linalg.norm(A, axis=1, keepdims=True)) @ T.T
    assert np.abs(want - new.values(A)).max() < 1e-5


def test_clap_concepts_derive_from_the_clap_store(tmp_path):
    from octantvector.generation.store import Store  # what derive itself reads and writes
    A = np.random.default_rng(4).standard_normal((3, 512)).astype(np.float32)
    src = Store(plug.Clap(), tmp_path / "store")
    for i, v in enumerate(A):
        src.add(f"{i:040x}", v)
    src.flush()
    new = plug.ClapConcepts()
    new.values = lambda X, device="cpu": X[:, :58]  # the store plumbing only (values: the test above)
    assert new.derive(tmp_path / "store") == 3
    hs, S = Store(new, tmp_path / "store").load()
    assert sorted(hs) == [f"{i:040x}" for i in range(3)] and np.allclose(S, A[[int(h, 16) for h in hs], :58])
    assert new.derive(tmp_path / "store") == 3  # nothing new: not scored again
