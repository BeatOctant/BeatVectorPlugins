"""Whisper large-v3 view for OctantVector, through octantvector.plugin (PLUGIN_API 0.3).

  whisper_language  the languages sung in the song (role language): over Whisper's 100 language tokens, the share of
                    the sung windows in each language, L2-normed (norm "none"). Two maps sung in one same language
                    have cosine 1, in different languages 0; a song partly in a second language sits in between. A
                    song with no sung window (no singing, or Whisper unsure) has no vector.

The whole song in ceil(length / CLIP_S) windows of CLIP_S seconds, spread evenly from its start to its end (so they
overlap a little, evenly, and no stretch of the song is counted twice as much as the rest; a song shorter than CLIP_S
is one window), each resampled to 16 kHz. A window's language probabilities are the softmax, over the language
tokens only, of the decoder's first prediction after <|startoftranscript|> (no generation). A window is not sung when
the entropy of those probabilities is above MAX_ENTROPY (nats; the 97th percentile over songs tagged with a sung
genre), when the best language is one of NO_SINGING (where Whisper puts instrumental songs) or when its probability is
below MIN_PROB. The rule was found in the spike of 2026-10-19 on one window from 30% of the song, which lands on intros and
breaks: of the songs the whole song finds a language for, one window finds 82% (OctantSpecs octantvector/experiments.md).

The model is read from models_dir only (octantharvest models).
"""
import json

import numpy as np

from octantvector.plugin import AudioEncoder, Model, register, resample

OCTANTVECTOR_PLUGIN_API = "0.3"

WHISPER = Model("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1",
                ("config.json", "generation_config.json", "preprocessor_config.json", "model.safetensors"))
CLIP_S = 30.0
GPU_WINDOWS = 16  # windows per forward pass, whatever the songs' lengths
MAX_ENTROPY = 1.9513493776321411
MIN_PROB = 0.5
NO_SINGING = ("ba", "bo", "cy", "eu", "haw", "jw", "km", "la", "ln", "lo", "mi", "my", "nn", "sa", "si", "sn", "tk",
              "yo")


def windows(y, sr):
    """The song as ceil(n / L) clips of L = CLIP_S seconds, the first at its start and the last at its end, the others
    evenly between (a song no longer than CLIP_S is one clip, the whole song)."""
    n, L = len(y), int(CLIP_S * sr)
    if n <= L:
        return [y]
    k = -(-n // L)
    return [y[s:s + L] for s in np.linspace(0, n - L, k).round().astype(int)]


def language(p, codes):
    """The index of the language sung, from the probabilities `p` over `codes`, or None (no clear language)."""
    p = np.asarray(p, np.float64)
    top = int(np.argmax(p))
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    if entropy > MAX_ENTROPY or codes[top] in NO_SINGING or p[top] < MIN_PROB:
        return None
    return top


def shares(tops, dim):
    """The L2-normed count of each language over the sung windows (`tops`: a language index or None per window), or
    None when no window is sung."""
    sung = [t for t in tops if t is not None]
    if not sung:
        return None
    v = np.bincount(sung, minlength=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@register
class WhisperLanguage(AudioEncoder):
    id = "whisper_language"
    dim = 100
    sr = 16000
    norm = "none"
    role = "language"
    batch = 4  # songs, whose windows go to the GPU GPU_WINDOWS at a time
    models = (WHISPER,)
    credits = ("Whisper large-v3 (OpenAI), https://huggingface.co/openai/whisper-large-v3, Apache-2.0 according to "
               "the model card",)

    def params(self):
        return {"clip_s": CLIP_S, "windows": "whole song", "max_entropy": MAX_ENTROPY, "min_prob": MIN_PROB,
                "no_singing": list(NO_SINGING), "sr": self.sr}

    def tokens(self):
        """[(language code, token id)] of Whisper's language tokens, by token id."""
        langs = json.loads((self.model_dir() / "generation_config.json").read_text())["lang_to_id"]
        return sorted(((tok.strip("<|>"), int(i)) for tok, i in langs.items()), key=lambda t: t[1])

    def setup_worker(self):
        from transformers import WhisperFeatureExtractor
        self.proc = WhisperFeatureExtractor.from_pretrained(self.model_dir(), local_files_only=True)

    def prep(self, y, sr):
        """-> (windows, mel bins, frames) float16: the windows' log-mel features (float16 to keep the queue small; the
        model runs in float16 on the GPU)."""
        L = int(CLIP_S * self.sr)
        clips = [resample(w, sr, self.sr).astype(np.float32) for w in windows(y, sr)]
        clips = [np.pad(c, (0, max(0, L - len(c)))) for c in clips]
        return self.proc(clips, sampling_rate=self.sr, return_tensors="np").input_features.astype(np.float16)

    def setup_model(self, device):
        import torch
        from transformers import WhisperForConditionalGeneration
        folder = self.model_dir()
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            folder, local_files_only=True, torch_dtype=self.dtype).to(device).eval()
        gen = json.loads((folder / "generation_config.json").read_text())
        self.start = int(gen.get("decoder_start_token_id", 50258))
        toks = self.tokens()
        if len(toks) != self.dim:
            raise ValueError(f"{WHISPER.key} has {len(toks)} language tokens, {self.id} is defined for {self.dim}")
        self.codes = [c for c, _ in toks]
        self.lang_ids = [i for _, i in toks]

    def embed(self, prepped):
        return self.embed_many([prepped])[0]

    def embed_many(self, prepped):
        import torch
        allw, tops = np.concatenate(prepped), []
        with torch.inference_mode():
            for i in range(0, len(allw), GPU_WINDOWS):
                x = torch.from_numpy(allw[i:i + GPU_WINDOWS]).to(self.device, self.dtype)
                start = torch.full((len(x), 1), self.start, dtype=torch.long, device=self.device)
                logits = self.model(input_features=x, decoder_input_ids=start).logits[:, -1, self.lang_ids]
                tops += [language(row, self.codes) for row in torch.softmax(logits.float(), -1).cpu().numpy()]
        out, i = [], 0
        for w in prepped:
            out.append(shares(tops[i:i + len(w)], self.dim))
            i += len(w)
        return out
