"""octantvector-clap: registration through the entry point and the declarations (no torch needed)."""
import json
from pathlib import Path
import subprocess
import sys
from importlib.metadata import entry_points

import octantvector_clap as plug
from octantvector import plugin

REV = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
IDS = ("clap_htsat_unfused", "vocal_clap_htsat_unfused", "clap_concepts")


def test_registered_through_the_entry_point():
    (ep,) = [e for e in entry_points(group="octantvector.encoders") if e.name == "clap"]
    assert ep.value == "octantvector_clap"
    assert plug.OCTANTVECTOR_PLUGIN_API == plugin.PLUGIN_API
    # a fresh process that never imports the plugin: OctantVector finds it through the entry point (contract 1)
    out = subprocess.run([sys.executable, "-c", "import json; from octantvector import generation as g; "
                          f"print(json.dumps([g.models([i]) for i in {list(IDS)!r}]))"],
                         capture_output=True, text=True, check=True).stdout
    clap, vocal, concepts = json.loads(out.splitlines()[-1])
    assert clap == concepts == [{"repo": "laion/clap-htsat-unfused", "revision": REV, "files": list(plug.CLAP.files),
                                 "path": f"hf/laion/clap-htsat-unfused/{REV}"}]
    assert [m["repo"] for m in vocal] == ["adefossez/HTDemucs", "laion/clap-htsat-unfused"]


def test_declarations():
    clap, vocal, concepts = plug.Clap(), plug.VocalClap(), plug.ClapConcepts()
    assert (clap.id, clap.dim, clap.role_name(), clap.version) == ("clap_htsat_unfused", 512, "music", "1")
    assert (vocal.id, vocal.dim, vocal.role_name(), vocal.batch, vocal.clip_s) == (
        "vocal_clap_htsat_unfused", 512, "vocal", 8, 7.8)
    assert (concepts.id, concepts.dim, concepts.role_name(), concepts.kind) == ("clap_concepts", 58, "genre", "derived")
    assert concepts.names == list(plug.CONCEPTS) and len(set(concepts.names)) == 58
    assert clap.definition()["params"]["models"] == [f"laion/clap-htsat-unfused@{REV}"]
    assert [m.repo for m in concepts.required_models()] == ["laion/clap-htsat-unfused"]


def test_the_plugin_reads_only_octantvector_plugin():
    text = Path(plug.__file__).read_text()
    assert "from octantvector.plugin import" in text and "octantvector.generation" not in text


def test_every_view_credits_the_model_and_a_vocal_view_also_demucs():
    from octantvector.generation.encoders import get
    for i in ("clap_htsat_unfused", "vocal_clap_htsat_unfused", "clap_concepts",):
        got = get(i).all_credits()
        assert got[0].startswith("CLAP (LAION)"), got
        assert any(c.startswith("Demucs") for c in got) == i.startswith("vocal_"), got
