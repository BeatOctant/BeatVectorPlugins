"""MERT-v1-95M views for OctantVector, through octantvector.plugin (PLUGIN_API 0.3).

  mert_v1_95m        the mix: time-mean of the outputs of encoder layers 4-8, averaged over the layers and 3 clips of 10 s
  vocal_mert_v1_95m  the same on the clips' vocal stem (role vocal): singing style and timbre

The model is built from the copy of its code in this package (modeling_MERT.py, configuration_MERT.py; NOTICE), never
from code in the model folder, and its weights are read from models_dir only (octantharvest models). Hidden states are
taken with hooks on the layers (OctantSpecs octantvector/encoders.md explains why).
"""
from octantvector.plugin import HiddenMean, Model, register

OCTANTVECTOR_PLUGIN_API = "0.3"

MERT = Model("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
             ("config.json", "preprocessor_config.json", "pytorch_model.bin"))


@register
class Mert95(HiddenMean):
    id = "mert_v1_95m"
    dim = 768
    sr = 24000
    layers = (4, 5, 6, 7, 8)
    models = (MERT,)
    credits = ("MERT-v1-95M (m-a-p), https://huggingface.co/m-a-p/MERT-v1-95M, weights and the copy of its code in "
               "octantvector-mert under CC BY-NC 4.0",)

    def load_processor(self, folder):
        from transformers import Wav2Vec2FeatureExtractor
        return Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)

    def load_model(self, folder, device):
        from .modeling_MERT import MERTModel
        return MERTModel.from_pretrained(folder, local_files_only=True).to(device).eval()


@register
class VocalMert95(Mert95):
    id = "vocal_mert_v1_95m"
    stem = "vocal"
