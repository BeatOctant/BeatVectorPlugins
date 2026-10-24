"""LAION CLAP (HTS-AT, unfused) views for OctantVector, through octantvector.plugin (PLUGIN_API 0.3).

  clap_htsat_unfused        the mix: mean of the L2-normed embeddings of 3 clips of 10 s
  vocal_clap_htsat_unfused  the same model on the clips' vocal stem (role vocal): the kind of voice, also reachable
                            by words
  clap_concepts             derived from clap_htsat_unfused: cosine with the text embedding of each concept (role
                            genre: each dimension is one word, so a hand filters by it instead of using it as a
                            similarity axis; OctantSpecs contracts/vector-db.md, 役割名)

Why not larger_clap_music: OctantSpecs octantvector/encoders.md. The model is read from models_dir only (octantharvest models).
"""
import numpy as np

from octantvector.plugin import ClipAudioEncoder, DerivedEncoder, Model, register

OCTANTVECTOR_PLUGIN_API = "0.3"

CLAP = Model("laion/clap-htsat-unfused", "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
             ("config.json", "preprocessor_config.json", "pytorch_model.bin", "merges.txt", "vocab.json",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"))

CONCEPTS = (
    [f"{g} music" for g in ["pop", "rock", "heavy metal", "punk", "electronic dance", "dubstep", "drum and bass",
                            "hardcore techno", "speedcore", "trance", "house", "techno", "hip hop", "jazz", "swing",
                            "classical orchestral", "acoustic folk", "funk disco", "j-pop", "j-rock", "k-pop",
                            "anime theme song", "vocaloid", "video game", "chiptune", "lo-fi", "ambient", "nightcore",
                            "meme", "film soundtrack"]]
    + ["happy upbeat music", "sad melancholic music", "aggressive intense music", "calm relaxing music",
       "epic cinematic music", "cute kawaii music", "dark music", "romantic music"]
    + ["female vocals", "male vocals", "instrumental music without vocals", "rap vocals", "choir",
       "autotuned vocals", "japanese lyrics", "english lyrics", "korean lyrics"]
    + ["electric guitar", "piano", "synthesizer", "string section", "heavy drums", "bass drop", "brass section"]
    + ["fast tempo", "slow tempo", "high energy", "low energy"])


def load_model(folder, device):
    from transformers import ClapModel
    return ClapModel.from_pretrained(folder, local_files_only=True).to(device).eval()


def load_processor(folder):
    from transformers import ClapProcessor
    return ClapProcessor.from_pretrained(folder, local_files_only=True)


@register
class Clap(ClipAudioEncoder):
    id = "clap_htsat_unfused"
    dim = 512
    sr = 48000
    l2_clips = True
    models = (CLAP,)
    credits = ("CLAP (LAION), https://github.com/LAION-AI/CLAP, model laion/clap-htsat-unfused, Apache-2.0",)

    def load_processor(self, folder):
        return load_processor(folder)

    def load_model(self, folder, device):
        return load_model(folder, device)

    def features(self, clips):
        f = self.proc(audio=clips, sampling_rate=self.sr, return_tensors="np")
        return f["input_features"].astype(np.float32), f["is_longer"]

    def clip_vectors(self, x):
        import torch
        feats, longer = x
        with torch.inference_mode():
            a = self.model.get_audio_features(input_features=torch.from_numpy(feats).to(self.device),
                                              is_longer=torch.from_numpy(longer).to(self.device))
            return getattr(a, "pooler_output", a).float().cpu().numpy()


@register
class VocalClap(Clap):
    id = "vocal_clap_htsat_unfused"
    stem = "vocal"


@register
class ClapConcepts(DerivedEncoder):
    id = "clap_concepts"
    inputs = ("view:clap_htsat_unfused",)
    role = "genre"
    models = (CLAP,)  # the text side of the same model
    names = list(CONCEPTS)  # the DB lists them as the view's genres
    dim = len(CONCEPTS)

    def params(self):
        return {"source": self.source, "concepts": self.names,
                "value": "cosine(L2(audio clap), L2(text clap)) per concept"}

    def text_embeddings(self, device="cpu"):
        """(concepts, dim) L2-normed CLAP text embeddings of the concepts."""
        import torch
        folder = self.model_dir()
        model, proc = load_model(folder, device), load_processor(folder)
        with torch.inference_mode():
            t = model.get_text_features(**proc(text=self.names, return_tensors="pt", padding=True).to(device))
            t = getattr(t, "pooler_output", t)
            return torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy()

    def values(self, A, device="cpu"):
        """(maps, dim) CLAP audio vectors -> (maps, concepts) cosines."""
        return (A / np.linalg.norm(A, axis=1, keepdims=True)) @ self.text_embeddings(device).T
