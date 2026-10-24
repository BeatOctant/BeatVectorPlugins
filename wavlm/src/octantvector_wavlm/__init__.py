"""WavLM speaker-verification view for OctantVector, through octantvector.plugin (PLUGIN_API 0.3).

  vocal_wavlm_sv  x-vector of the clips' vocal stem (role vocal): the voice itself; mean of the L2-normed clip vectors

The model is read from models_dir only (octantharvest models).
"""
import numpy as np

from octantvector.plugin import ClipAudioEncoder, Model, register

OCTANTVECTOR_PLUGIN_API = "0.3"

WAVLM = Model("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
              ("config.json", "preprocessor_config.json", "pytorch_model.bin"))


@register
class VocalWavlm(ClipAudioEncoder):
    id = "vocal_wavlm_sv"
    dim = 512
    sr = 16000
    l2_clips = True
    stem = "vocal"
    models = (WAVLM,)
    credits = ("WavLM-Base-Plus-SV (Microsoft), https://huggingface.co/microsoft/wavlm-base-plus-sv, under CC BY-SA 3.0 "
               "(the model card's licence link, https://github.com/microsoft/UniSpeech/blob/main/LICENSE)",)

    def load_processor(self, folder):
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_pretrained(folder, local_files_only=True)

    def load_model(self, folder, device):
        from transformers import WavLMForXVector
        return WavLMForXVector.from_pretrained(folder, local_files_only=True).to(device).eval()

    def features(self, clips):
        f = self.proc(clips, sampling_rate=self.sr, return_tensors="np", padding=True)
        return {k: v.astype(np.float32) if k == "input_values" else v for k, v in f.items()}

    def clip_vectors(self, x):
        import torch
        with torch.inference_mode():
            e = self.model(**{k: torch.from_numpy(v).to(self.device) for k, v in x.items()}).embeddings
            return e.float().cpu().numpy()
