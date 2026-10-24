"""octantvector-siglip: registration, the declarations, the picture shrinking, and one run of a tiny randomly initialised
SigLIP from a temporary models_dir (no network, no real weights)."""
import io
import json
import subprocess
import sys
from importlib.metadata import entry_points

import numpy as np
import pytest

import octantvector_siglip as plug
from octantvector import plugin

REV = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
Image = pytest.importorskip("PIL.Image")


def test_registered_through_the_entry_point():
    (ep,) = [e for e in entry_points(group="octantvector.encoders") if e.name == "siglip"]
    assert ep.value == "octantvector_siglip"
    assert plug.OCTANTVECTOR_PLUGIN_API == plugin.PLUGIN_API
    out = subprocess.run([sys.executable, "-c", "import json; from octantvector import generation as g; "
                          "print(json.dumps(g.models(['siglip_cover'])))"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.splitlines()[-1]) == [
        {"repo": "google/siglip-so400m-patch14-384", "revision": REV,
         "files": ["config.json", "model.safetensors", "preprocessor_config.json"],
         "path": f"hf/google/siglip-so400m-patch14-384/{REV}"}]


def test_declarations():
    e = plug.SiglipCover()
    assert (e.id, e.dim, e.norm, e.kind, e.role_name(), e.rows, e.inputs) == (
        "siglip_cover", 1152, "none", "image", "cover", "maps", ("cover",))
    assert e.definition()["params"] == {"shrink": 768, "models": [f"google/siglip-so400m-patch14-384@{REV}"]}
    assert e.all_credits()[0].startswith("SigLIP so400m")


def _png(w, h, mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), 128).save(buf, fmt)
    return buf.getvalue()


def test_a_picture_is_rgb_and_fits_768_px_keeping_its_shape():
    assert plug.picture(_png(2000, 1000)).size == (768, 384)
    assert plug.picture(_png(300, 200, "L", "JPEG")).mode == "RGB"
    assert plug.picture(_png(300, 200, "RGBA")).size == (300, 200)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel
    torch.manual_seed(0)
    d = tmp_path / "models" / plug.SIGLIP.path
    d.mkdir(parents=True)
    cfg = SiglipConfig(text_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                                    "intermediate_size": 32, "vocab_size": 64},
                       vision_config={"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                                      "intermediate_size": 32, "image_size": 384, "patch_size": 14})
    SiglipModel(cfg).eval().save_pretrained(d)
    SiglipImageProcessor(size={"height": 384, "width": 384}).save_pretrained(d)
    assert sorted(p.name for p in d.iterdir()) == sorted(plug.SIGLIP.files)
    monkeypatch.setenv("BV_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("BV_MODELS_DIR", str(tmp_path / "models"))
    return d


def test_a_tiny_siglip_gives_unit_vectors_and_the_same_picture_the_same_vector(models_dir):
    e = plug.SiglipCover()
    e.setup_worker()
    rng = np.random.default_rng(0)
    pics = []
    for _ in range(2):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (500, 400, 3), np.uint8)).save(buf, "PNG")
        pics.append(buf.getvalue())
    x = [e.prep(p) for p in pics + pics[:1]]
    assert x[0].shape == (3, 384, 384) and x[0].dtype == np.float16
    e.setup_model("cpu")
    a, b, a2 = e.embed_many(x)
    assert a.shape == (16,) and np.isclose(np.linalg.norm(a), 1, atol=1e-5)  # the tiny model's projection size
    assert np.allclose(a, a2, atol=1e-6) and not np.allclose(a, b, atol=1e-6)
    assert np.allclose(e.embed(x[1]), b, atol=1e-6)
