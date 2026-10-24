"""octantvector-mert: registration through the entry point, the declarations and the copied model code (no torch
needed)."""
import hashlib
import json
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import octantvector_mert as plug
from octantvector import plugin

REV = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
SRC = Path(plug.__file__).parent
HEADER_LINES = 5  # the comment block added above the copied content


def test_registered_through_the_entry_point():
    (ep,) = [e for e in entry_points(group="octantvector.encoders") if e.name == "mert"]
    assert ep.value == "octantvector_mert"
    assert plug.OCTANTVECTOR_PLUGIN_API == plugin.PLUGIN_API
    # a fresh process that never imports the plugin: OctantVector finds it through the entry point (contract 1)
    out = subprocess.run([sys.executable, "-c", "import json; from octantvector import generation as g; print(json.dumps("
                          "[g.models(['mert_v1_95m']), g.models(['vocal_mert_v1_95m'])]))"],
                         capture_output=True, text=True, check=True).stdout
    mix, vocal = json.loads(out.splitlines()[-1])
    assert mix == [{"repo": "m-a-p/MERT-v1-95M", "revision": REV,
                    "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
                    "path": f"hf/m-a-p/MERT-v1-95M/{REV}"}]
    assert [m["repo"] for m in vocal] == ["adefossez/HTDemucs", "m-a-p/MERT-v1-95M"]


def test_declarations():
    mix, vocal = plug.Mert95(), plug.VocalMert95()
    assert (mix.id, mix.dim, mix.sr, mix.layers, mix.role_name(), mix.l2_clips) == (
        "mert_v1_95m", 768, 24000, (4, 5, 6, 7, 8), "music", False)
    assert (vocal.id, vocal.dim, vocal.role_name(), vocal.batch, vocal.clip_s, vocal.layers) == (
        "vocal_mert_v1_95m", 768, "vocal", 8, 7.8, (4, 5, 6, 7, 8))
    assert mix.definition()["params"]["models"] == [f"m-a-p/MERT-v1-95M@{REV}"]


def test_the_model_code_is_the_pinned_upstream_copy():
    """Byte-for-byte below the added header (sha256 of the files at the pinned revision)."""
    want = {"modeling_MERT.py": "6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c",
            "configuration_MERT.py": "ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03"}
    for name, sha in want.items():
        lines = (SRC / name).read_bytes().split(b"\n")
        assert REV.encode() in b"\n".join(lines[:HEADER_LINES])
        assert hashlib.sha256(b"\n".join(lines[HEADER_LINES:])).hexdigest() == sha


def test_the_plugin_reads_only_octantvector_plugin():
    text = (SRC / "__init__.py").read_text()
    assert "from octantvector.plugin import" in text and "octantvector.generation" not in text
    assert "trust_remote_code" not in text


def test_every_view_credits_the_model_and_a_vocal_view_also_demucs():
    from octantvector.generation.encoders import get
    for i in ("mert_v1_95m", "vocal_mert_v1_95m",):
        got = get(i).all_credits()
        assert got[0].startswith("MERT-v1-95M"), got
        assert any(c.startswith("Demucs") for c in got) == i.startswith("vocal_"), got
