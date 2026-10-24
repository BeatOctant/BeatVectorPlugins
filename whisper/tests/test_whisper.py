"""octantvector-whisper: registration, the declarations, the window and the language rule, and one run of a tiny randomly
initialised Whisper from a temporary models_dir (no network, no real weights)."""
import json
import subprocess
import sys
from importlib.metadata import entry_points

import numpy as np
import pytest

import octantvector_whisper as plug
from octantvector import plugin

REV = "06f233fe06e710322aca913c1bc4249a0d71fce1"


def test_registered_through_the_entry_point():
    (ep,) = [e for e in entry_points(group="octantvector.encoders") if e.name == "whisper"]
    assert ep.value == "octantvector_whisper"
    assert plug.OCTANTVECTOR_PLUGIN_API == plugin.PLUGIN_API
    out = subprocess.run([sys.executable, "-c", "import json; from octantvector import generation as g; "
                          "print(json.dumps(g.models(['whisper_language'])))"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.splitlines()[-1]) == [
        {"repo": "openai/whisper-large-v3", "revision": REV,
         "files": ["config.json", "generation_config.json", "preprocessor_config.json", "model.safetensors"],
         "path": f"hf/openai/whisper-large-v3/{REV}"}]


def test_declarations():
    e = plug.WhisperLanguage()
    assert (e.id, e.dim, e.sr, e.norm, e.kind, e.role_name(), e.rows) == (
        "whisper_language", 100, 16000, "none", "audio", "language", "maps")
    assert e.definition()["params"]["models"] == [f"openai/whisper-large-v3@{REV}"]
    assert e.definition()["params"]["max_entropy"] == 1.9513493776321411
    assert e.all_credits()[0].startswith("Whisper large-v3")


def test_the_windows_cover_the_whole_song_evenly():
    sr = 10
    y = np.arange(900, dtype=np.float32)  # 90 s: three windows side by side
    assert [(w[0], len(w)) for w in plug.windows(y, sr)] == [(0, 300), (300, 300), (600, 300)]
    y = np.arange(1000, dtype=np.float32)  # 100 s: four windows, start to end, overlapping evenly
    assert [w[0] for w in plug.windows(y, sr)] == [0, 233, 467, 700]
    y = np.arange(901, dtype=np.float32)  # just over 90 s: no window nearly the same as the one before it
    assert [w[0] for w in plug.windows(y, sr)] == [0, 200, 401, 601]
    short = np.arange(400, dtype=np.float32)  # 40 s
    assert [w[0] for w in plug.windows(short, sr)] == [0, 100]
    tiny = np.arange(100, dtype=np.float32)  # 10 s: one window, the whole song
    assert [len(w) for w in plug.windows(tiny, sr)] == [100]


def test_the_vector_is_the_share_of_the_sung_windows_in_each_language():
    assert plug.shares([None, None], 4) is None
    assert plug.shares([1, None, 1, 1], 4).tolist() == [0, 1, 0, 0]
    v = plug.shares([1, 1, 1, 3], 4)  # three windows in one language, one in another
    assert np.allclose(v, np.array([0, 3, 0, 1]) / np.sqrt(10)) and np.isclose(v @ plug.shares([1], 4), 3 / np.sqrt(10))


def test_a_song_has_a_language_only_when_whisper_is_sure_and_it_is_not_a_no_singing_language():
    codes = ["en", "ja", "km", "ko"] + [f"x{i}" for i in range(96)]

    def p(**share):
        v = np.full(100, (1 - sum(share.values())) / (100 - len(share)))
        for c, s in share.items():
            v[codes.index(c)] = s
        return v
    assert plug.language(p(ja=0.9), codes) == 1
    assert plug.language(p(km=0.9), codes) is None  # where instrumental songs go
    assert plug.language(p(ja=0.45, ko=0.44), codes) is None  # unsure
    assert plug.language(p(ja=0.55), codes) is None  # the rest spread wide: entropy above the threshold
    assert plug.language(p(ja=0.6, en=0.4 - 1e-9), codes) == 1


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A tiny random Whisper under the pinned path, with the real large-v3 language tokens (100, from 50259)."""
    from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForConditionalGeneration
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    torch.manual_seed(0)
    d = tmp_path / "models" / plug.WHISPER.path
    d.mkdir(parents=True)
    cfg = WhisperConfig(vocab_size=51866, num_mel_bins=128, d_model=16, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                        max_source_positions=1500, max_target_positions=8, decoder_start_token_id=50258)
    WhisperForConditionalGeneration(cfg).eval().save_pretrained(d)
    gen = json.loads((d / "generation_config.json").read_text())
    gen["lang_to_id"] = {f"<|{c}|>": 50259 + n for n, c in enumerate(LANGUAGES)}
    (d / "generation_config.json").write_text(json.dumps(gen))
    WhisperFeatureExtractor(feature_size=128).save_pretrained(d)
    assert sorted(p.name for p in d.iterdir()) == sorted(plug.WHISPER.files)
    monkeypatch.setenv("BV_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("BV_MODELS_DIR", str(tmp_path / "models"))
    return d


def test_a_tiny_whisper_gives_a_one_hot_of_the_language_or_nothing(models_dir):
    e = plug.WhisperLanguage()
    e.setup_worker()
    sr = 22050
    t = np.arange(sr * 40) / sr
    x = e.prep((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)  # 40 s: two distinct windows
    assert x.shape == (2, 128, 3000) and x.dtype == np.float16
    e.setup_model("cpu")
    assert len(e.codes) == 100 and e.codes[:3] == ["en", "zh", "de"] and e.lang_ids[0] == 50259

    class Logits:  # the decoder's prediction per window: Japanese, English, then nothing clear for the second song
        def __init__(self, n):
            self.logits = torch.zeros(n, 1, 51866)
            self.logits[0, 0, 50259 + e.codes.index("ja")] = 20.0
            self.logits[1, 0, 50259 + e.codes.index("en")] = 20.0

    e.model = lambda input_features, decoder_input_ids: (
        Logits(len(input_features)) if decoder_input_ids.tolist() == [[50258]] * len(input_features) else None)
    both, none = e.embed_many([x, x])
    assert none is None and both.shape == (100,) and np.isclose(both @ both, 1)
    assert np.isclose(both[e.codes.index("ja")], both[e.codes.index("en")]) and np.count_nonzero(both) == 2


def test_the_tiny_model_runs_end_to_end(models_dir):
    e = plug.WhisperLanguage()
    e.setup_worker()
    e.setup_model("cpu")
    y = np.random.default_rng(1).standard_normal(16000 * 5).astype(np.float32)
    out = e.embed(e.prep(y, 16000))  # a random model is unsure: no language
    assert out is None or (out.shape == (100,) and np.isclose(out @ out, 1))


def test_the_windows_go_to_the_model_a_fixed_number_at_a_time(models_dir, monkeypatch):
    e = plug.WhisperLanguage()
    e.setup_worker()
    e.setup_model("cpu")
    monkeypatch.setattr(plug, "GPU_WINDOWS", 3)
    sizes = []

    def model(input_features, decoder_input_ids):
        sizes.append(len(input_features))
        logits = torch.zeros(len(input_features), 1, 51866)
        logits[:, 0, 50259 + e.codes.index("ja")] = 20.0
        return type("Out", (), {"logits": logits})
    e.model = model
    songs = [np.zeros((2, 128, 3000), np.float16), np.zeros((5, 128, 3000), np.float16)]
    a, b = e.embed_many(songs)
    assert sizes == [3, 3, 1] and a[e.codes.index("ja")] == 1 and np.array_equal(a, b)
