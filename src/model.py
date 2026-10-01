"""Thin wrappers around pretrained video models. No training anywhere.

Every wrapper exposes:
    .name    : str
    .labels  : list[str]          (class names, index-aligned with probabilities)
    .predict(clips) -> (probs [N, C], embeds [N, D])
"""
from __future__ import annotations

import numpy as np

DEFAULT_XCLIP_LABELS = [
    "cooking", "cutting vegetables", "stirring food", "washing dishes", "cleaning",
    "walking", "running", "playing sports", "dancing", "talking to camera",
    "using a phone", "typing on a keyboard", "driving", "idle or no activity",
]

def _pick_device(device: str | None) -> str:
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")

class XCLIPModel:
    """Enhanced X-CLIP zero-shot video-text matching with batched inference and FP16 support."""

    def __init__(self, labels: list[str], model_name: str = "microsoft/xclip-base-patch32",
                 template: str = "a video of {}", device: str | None = None):
        import torch
        from transformers import XCLIPModel as HFXCLIP, XCLIPProcessor

        self.torch = torch
        self.device = _pick_device(device)
        self.fp16 = self.device == "cuda"
        
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self.model = HFXCLIP.from_pretrained(model_name).to(self.device).eval()
        
        if self.fp16:
            self.model = self.model.half()
            
        self.num_frames = self.model.config.vision_config.num_frames
        self.labels = list(labels)
        self.prompts = [template.format(l) for l in self.labels]
        self.name = "xclip-zeroshot-batched"

    def predict(self, clips):
        torch = self.torch
        sampled_clips = []
        
        # 1. Pre-sample frames for all clips in the batch
        for c in clips:
            idx = np.linspace(0, len(c) - 1, self.num_frames).round().astype(int)
            sampled_clips.append(list(c[idx]))

        # 2. Process the entire batch in a single processor call
        inputs = self.processor(text=self.prompts, videos=sampled_clips,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 3. Cast image tensors to FP16 on GPU
        if self.fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        # 4. Single batched forward pass
        with torch.no_grad():
            out = self.model(**inputs)
            
        probs = out.logits_per_video.softmax(-1).cpu().numpy()
        embeds = out.video_embeds.cpu().numpy()
        
        return probs, embeds

def load_model(labels: list[str] | None = None, device: str | None = None):
    """Loads the X-CLIP model exclusively."""
    return XCLIPModel(labels or DEFAULT_XCLIP_LABELS, device=device)