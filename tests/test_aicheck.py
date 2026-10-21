import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from aicheck import files, provenance
from aicheck.pipeline import analyze, noisy_or

CFG = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text())


def _noise_img(size=(640, 480)):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def run(path, tmp_path):
    return analyze(path, CFG, [], tmp_path / "work")


def test_plain_image_is_low(tmp_path):
    p = tmp_path / "plain.jpg"
    _noise_img().save(p)
    rep = run(p, tmp_path)
    assert rep["band"] == "low" and rep["ai_likelihood"] == 0.0 and rep["file_type"] == "jpeg"


def test_png_with_sd_parameters_is_high(tmp_path):
    p = tmp_path / "gen.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1")
    _noise_img().save(p, pnginfo=meta)
    rep = run(p, tmp_path)
    assert rep["band"] == "high" and any("generation parameters" in r for r in rep["reasons"])


def test_iptc_ai_marker_is_high(tmp_path):
    p = tmp_path / "tagged.jpg"
    _noise_img().save(p)
    p.write_bytes(p.read_bytes() + b"<Iptc4xmpExt:DigitalSourceType>trainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>")
    rep = run(p, tmp_path)
    assert rep["band"] == "high"


def test_c2pa_without_ai_hint_is_only_context(tmp_path):
    p = tmp_path / "c2pa.jpg"
    _noise_img().save(p)
    p.write_bytes(p.read_bytes() + b"jumbf c2pa Leica Camera AG")
    rep = run(p, tmp_path)
    assert rep["ai_likelihood"] == 0.0 and any("C2PA" in c for c in rep["context"])


def test_wrong_extension_flagged_as_other_signal(tmp_path):
    p = tmp_path / "photo.pdf"
    _noise_img().save(p, format="JPEG")
    rep = run(p, tmp_path)
    assert rep["file_type"] == "jpeg" and any("extension" in s for s in rep["other_alteration_signals"])


def test_docx_embedded_image_is_extracted_and_checked(tmp_path):
    img = tmp_path / "x.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "Steps: 30, Sampler: DPM++, CFG scale: 5")
    _noise_img().save(img, pnginfo=meta)
    d = tmp_path / "report.docx"
    with zipfile.ZipFile(d, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w/>")
        z.write(img, "word/media/image1.png")
    assert files.sniff(d) == "docx"
    rep = run(d, tmp_path)
    assert rep["images_analyzed"] == 1 and rep["band"] == "high"


def test_pdf_with_embedded_generated_image(tmp_path):
    img = tmp_path / "x.png"
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "Steps: 30, Sampler: DPM++, CFG scale: 5")
    _noise_img().save(img, pnginfo=meta)
    pdf = tmp_path / "scan.pdf"
    Image.open(img).convert("RGB").save(pdf, "PDF")
    rep = run(pdf, tmp_path)
    assert rep["file_type"] == "pdf" and rep["images_analyzed"] >= 1


def test_unknown_file_does_not_crash(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"not a real file at all")
    rep = run(p, tmp_path)
    assert rep["file_type"] == "unknown" and rep["band"] == "low"


def test_generator_dimensions_is_weak(tmp_path):
    p = tmp_path / "g.png"
    _noise_img((1024, 1024)).save(p)
    rep = run(p, tmp_path)
    assert 0 < rep["ai_likelihood"] <= 0.15


def test_noisy_or():
    assert noisy_or([]) == 0.0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
