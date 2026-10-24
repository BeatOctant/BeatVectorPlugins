"""SigLIP so400m view for OctantVector, through octantvector.plugin (PLUGIN_API 0.3).

  siglip_cover  the map's cover image (role cover): SigLIP's image embedding (the attention-pooled head), L2-normed.
                The view keeps the plain cosine (norm "none"): covers that are the same picture stay near 1.

The cover is the file Info.dat names (else the first image in the zip; OctantVector's MapZip.cover_bytes). It is shrunk to
fit SHRINK px first (JPEG draft decoding, then a thumbnail), so very large covers cost little, then resized to 384 x 384
without keeping the aspect ratio, as SigLIP's image processor does. A map without a cover has no vector. Found in the
spike of 2026-10-19 (OctantSpecs octantvector/experiments.md).

The model is read from models_dir only (octantharvest models).
"""
import io

import numpy as np

from octantvector.plugin import ImageEncoder, Model, register

OCTANTVECTOR_PLUGIN_API = "0.3"

SIGLIP = Model("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
               ("config.json", "model.safetensors", "preprocessor_config.json"))
SHRINK = 768


def picture(data):
    """The image file's bytes as an RGB picture no larger than SHRINK px on either side."""
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.draft("RGB", (SHRINK, SHRINK))  # JPEG only: decode at a smaller scale
    img = img.convert("RGB")
    img.thumbnail((SHRINK, SHRINK))
    return img


@register
class SiglipCover(ImageEncoder):
    id = "siglip_cover"
    dim = 1152
    norm = "none"
    batch = 32
    models = (SIGLIP,)
    credits = ("SigLIP so400m-patch14-384 (Google), https://huggingface.co/google/siglip-so400m-patch14-384, "
               "Apache-2.0 according to the model card",)

    def params(self):
        return {"shrink": SHRINK}

    def setup_worker(self):
        from transformers import SiglipImageProcessor
        self.proc = SiglipImageProcessor.from_pretrained(self.model_dir(), local_files_only=True)

    def prep(self, data):
        return self.proc(picture(data), return_tensors="np").pixel_values[0].astype(np.float16)

    def setup_model(self, device):
        import torch
        from transformers import SiglipModel
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = SiglipModel.from_pretrained(self.model_dir(), local_files_only=True,
                                                 torch_dtype=self.dtype).to(device).eval()

    def embed(self, prepped):
        return self.embed_many([prepped])[0]

    def embed_many(self, prepped):
        import torch
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(prepped)).to(self.device, self.dtype)
            e = self.model.get_image_features(pixel_values=x)
            e = getattr(e, "pooler_output", e).float()  # some transformers versions return the model output
            e = torch.nn.functional.normalize(e, dim=-1)
            return list(e.cpu().numpy())
