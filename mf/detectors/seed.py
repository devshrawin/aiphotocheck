"""SEED document-tampering detector adapter (DINOv3 ViT-L + LoRA + EoMT head, MIT licence for SEED code/weights).

Source of truth: https://github.com/KahimWong/GenText-Forensics-3rd-Place (clone into third_party/SEED) and
its main.py `_predict_patch_batch` / `infer_from_image_paths`:
  * the model takes RAW 0..255 RGB float tensors (it divides by 255 and applies ImageNet mean/std itself),
  * images are cut into 512x512 tiles, short tiles padded with white (255),
  * per tile, image-level forged probability = softmax(image_logits)[:, 1]  (output tuple index 3).
We score a page as the max over tiles.

The EoMT constructor downloads `facebook/dinov3-vitl16-pretrain-lvd1689m`, which is a gated Hugging Face repo.
The SEED checkpoint already contains the full (LoRA-merged-in-structure) backbone weights, so we only need the
architecture: during construction `AutoModel.from_pretrained` is swapped for `AutoModel.from_config` with the
published ViT-L/16 hyper-parameters. Missing/unexpected keys are checked on load.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Detector, DetectorResult


def _starts(total: int, size: int) -> list[int]:
    if total <= size:
        return [0]
    n = -(-total // size)
    step = (total - size) / (n - 1)
    return [int(round(i * step)) for i in range(n)]


class SeedDetector(Detector):
    name = "seed"

    def __init__(self, cfg: dict, project_root: Path):
        self.repo = (project_root / cfg["repo_dir"]).resolve()
        self.hf_id, self.size = cfg["hf_id"], cfg.get("input_size", 512)
        self.max_side = cfg.get("max_side", 1024)
        self._model = None

    def available(self):
        if not (self.repo / "model").exists():
            return False, f"SEED repo not found at {self.repo} (clone the GitHub repo linked from the model card)"
        try:
            import torch  # noqa: F401
            self._load()
        except Exception as e:
            return False, f"could not load SEED: {str(e)[:200]}"
        return True, ""

    def _load(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModel, DINOv3ViTConfig

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from model.hf_wrapper import EoMTForTamperingDetection  # type: ignore

        vitl = DINOv3ViTConfig(
            hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
            patch_size=16, num_register_tokens=4, layerscale_value=1.0, rope_theta=100.0,
            query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, use_gated_mlp=False,
        )
        orig = AutoModel.from_pretrained
        AutoModel.from_pretrained = classmethod(lambda cls, *a, **k: AutoModel.from_config(vitl))  # type: ignore
        try:
            model = EoMTForTamperingDetection.from_pretrained(self.hf_id)
        finally:
            AutoModel.from_pretrained = orig
        self._model = model.eval()

    def score(self, image_path: Path, outdir: Path) -> DetectorResult:
        import torch

        self._load()
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        s = self.max_side / max(w, h)
        if s < 1:
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
            w, h = img.size
        S = self.size
        tiles = []
        for top in _starts(h, S):
            for left in _starts(w, S):
                p = np.asarray(img.crop((left, top, min(left + S, w), min(top + S, h))), np.uint8)
                pad = np.full((S, S, 3), 255, np.uint8)
                oy, ox = (S - p.shape[0]) // 2, (S - p.shape[1]) // 2
                pad[oy:oy + p.shape[0], ox:ox + p.shape[1]] = p
                tiles.append(pad)
        probs = []
        with torch.no_grad():
            for i in range(0, len(tiles), 4):
                x = torch.from_numpy(np.stack(tiles[i:i + 4])).permute(0, 3, 1, 2).float()
                out = self._model(x)
                probs += out[3].softmax(dim=1)[:, 1].tolist()
        return DetectorResult(self.name, float(max(probs)), extra={"tiles": len(tiles), "tile_probs": [round(p, 3) for p in probs]})
