"""Whole-image 'is this AI-generated?' classifiers from Hugging Face (Community Forensics ViT, SDXL detector).

These judge the whole image and do NOT localise edits. They were built for fully synthetic images,
so expect weak results on a real scan with one edited field. That is exactly what the evaluation is for.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import Detector, DetectorResult


class HFClassifierDetector(Detector):
    def __init__(self, name: str, model_id: str, fake_label_hints: list[str], onnx: dict | None = None):
        self.name, self.model_id, self.hints = name, model_id, [h.lower() for h in fake_label_hints]
        # onnx: set for repos that only ship ONNX weights (no pytorch/safetensors), which transformers cannot load.
        self.onnx = onnx
        self._pipe = None
        self._sess = None
        self._err = ""

    def available(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:
            return False, f"torch/transformers not installed ({e.__class__.__name__}); run setup.sh"
        try:
            self._load()
        except Exception as e:
            return False, f"could not load {self.model_id}: {str(e)[:160]}"
        return True, ""

    def _load(self):
        if self.onnx:
            return self._load_onnx()
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else -1
        self._pipe = pipeline("image-classification", model=self.model_id, device=device)

    def _load_onnx(self):
        if self._sess is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.model_id, self.onnx.get("file", "onnx/model.onnx"))
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def _score_onnx(self, img: Image.Image) -> DetectorResult:
        import numpy as np

        o = self.onnx
        resize, crop = o.get("resize", 440), o.get("crop", 384)
        img = img.resize((resize, resize), Image.BICUBIC)
        l = (resize - crop) // 2
        img = img.crop((l, l, l + crop, l + crop))
        x = (np.asarray(img, np.float32) / 255.0 - np.array(o["mean"], np.float32)) / np.array(o["std"], np.float32)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        logits = self._sess.run(None, {self._sess.get_inputs()[0].name: x})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        fake = float(p[o.get("fake_index", 1)])
        return DetectorResult(self.name, fake, extra={"logits": [round(float(v), 4) for v in logits]})

    def _is_fake(self, label: str) -> bool:
        l = label.lower()
        if "not" in l or "real" in l or "human" in l or "authentic" in l:
            return False
        return any(h in l for h in self.hints)

    def score(self, image_path: Path, outdir: Path) -> DetectorResult:
        self._load()
        img = Image.open(image_path).convert("RGB")
        if self.onnx:
            return self._score_onnx(img)
        preds = self._pipe(img, top_k=None)
        fake = sum(p["score"] for p in preds if self._is_fake(p["label"]))
        return DetectorResult(self.name, float(min(1.0, fake)), extra={"labels": {p["label"]: round(p["score"], 4) for p in preds}})
