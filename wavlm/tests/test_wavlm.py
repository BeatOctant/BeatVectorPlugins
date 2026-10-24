"""octantvector-wavlm: registration through the entry point and the declarations (no torch needed)."""
import json
import subprocess
import sys
from importlib.metadata import entry_points

import octantvector_wavlm as plug
from octantvector import plugin

REV = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"


def test_registered_through_the_entry_point():
    (ep,) = [e for e in entry_points(group="octantvector.encoders") if e.name == "wavlm"]
    assert ep.value == "octantvector_wavlm"
    assert plug.OCTANTVECTOR_PLUGIN_API == plugin.PLUGIN_API
    # a fresh process that never imports the plugin: OctantVector finds it through the entry point (contract 1)
    out = subprocess.run([sys.executable, "-c", "import json; from octantvector import generation as g; "
                          "print(json.dumps(g.models(['vocal_wavlm_sv'])))"],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out.splitlines()[-1]) == [
        {"repo": "adefossez/HTDemucs", "revision": "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003",
         "files": ["htdemucs.yaml", "955717e8.safetensors"],
         "path": "hf/adefossez/HTDemucs/cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"},
        {"repo": "microsoft/wavlm-base-plus-sv", "revision": REV,
         "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
         "path": f"hf/microsoft/wavlm-base-plus-sv/{REV}"}]


def test_declarations():
    e = plug.VocalWavlm()
    assert (e.id, e.dim, e.sr, e.l2_clips, e.role_name(), e.batch, e.clip_s) == (
        "vocal_wavlm_sv", 512, 16000, True, "vocal", 8, 7.8)
    assert e.definition()["params"]["models"] == [f"microsoft/wavlm-base-plus-sv@{REV}",
                                                  "adefossez/HTDemucs@cbc8a9b1a87023b7fd74e7b3412e6321c0eab003"]


def test_every_view_credits_the_model_and_a_vocal_view_also_demucs():
    from octantvector.generation.encoders import get
    for i in ("vocal_wavlm_sv",):
        got = get(i).all_credits()
        assert got[0].startswith("WavLM-Base-Plus-SV"), got
        assert any(c.startswith("Demucs") for c in got) == i.startswith("vocal_"), got
