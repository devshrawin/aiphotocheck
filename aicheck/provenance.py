"""Provenance and metadata checks: cheap, explainable, high precision when they fire, silent when metadata is stripped.

Every finding has a `kind`:
  ai    evidence of AI generation or AI editing (feeds the AI-likelihood score)
  edit  evidence of non-AI manipulation (reported separately, can raise the review band)
  info  context for the reviewer, weight 0
Weights are heuristics in [0, 1], not probabilities.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Present only in files produced or edited by generative tools.
STRONG_AI_MARKERS = [
    b"trainedalgorithmicmedia",            # IPTC DigitalSourceType for AI-generated
    b"compositewithtrainedalgorithmicmedia",  # IPTC: AI-edited composite
    b"midjourney", b"stable diffusion", b"dall-e", b"dall\xc2\xb7e", b"gpt-image", b"adobe firefly",
    b"nano banana", b"made with google ai", b"synthid",
]
WEAK_AI_MARKERS = [b"ideogram", b"imagen", b"firefly", b"leonardo.ai", b"flux", b"comfyui"]
# C2PA content credentials are also written by real cameras, so presence alone is not AI evidence.
C2PA_MARKERS = [b"c2pa", b"jumbf"]
C2PA_AI_HINTS = [b"openai", b"chatgpt", b"gemini", b"firefly", b"dall-e", b"midjourney", b"imagen"]
PNG_PROMPT_KEYS = {"parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata", "dream", "comment"}
EDITOR_MARKERS = [
    "photoshop", "gimp", "pixlr", "canva", "snapseed", "paint.net", "illustrator",
    "affinity", "lightroom", "photopea", "picsart", "facetune", "remini",
]
PDF_EDIT_TOOLS = [
    "ilovepdf", "smallpdf", "sejda", "pdf-xchange", "foxit", "nitro", "wondershare",
    "pdfescape", "sodapdf", "pdfelement", "acrobat", "pdffiller", "pdf24",
]


@dataclass
class Finding:
    id: str
    kind: str
    weight: float
    message: str

    def to_dict(self):
        return asdict(self)


def _parse_pdf_date(s) -> datetime | None:
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", str(s or ""))
    if not m:
        return None
    parts = [int(g) if g else d for g, d in zip(m.groups(), (1970, 1, 1, 0, 0, 0))]
    try:
        return datetime(*parts, tzinfo=timezone.utc)
    except ValueError:
        return None


def file_checks(path: Path, kind: str) -> list[Finding]:
    path = Path(path)
    out: list[Finding] = []
    ext = path.suffix.lower().lstrip(".")
    ext_kind = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "pdf": "pdf", "tif": "tiff", "tiff": "tiff", "webp": "webp",
                "gif": "gif", "bmp": "bmp", "docx": "docx", "pptx": "pptx", "xlsx": "xlsx"}.get(ext)
    if kind == "unknown":
        return [Finding("unknown_type", "info", 0.0, "File type could not be identified from its contents; only metadata checks ran.")]
    if ext_kind and ext_kind != kind:
        out.append(Finding("extension_mismatch", "edit", 0.3, f"File extension .{ext} does not match its actual content ({kind})."))

    raw = path.read_bytes()
    low = raw.lower()
    out += _marker_findings(low)
    if kind == "png":
        out += _png_prompt_findings(path)
    if kind == "pdf":
        out += _pdf_checks(path, raw)
    elif kind in ("jpeg", "png", "tiff", "webp"):
        out += _exif_checks(path)
    return out


def _marker_findings(low: bytes) -> list[Finding]:
    out = []
    for m in STRONG_AI_MARKERS:
        if m in low:
            out.append(Finding("ai_marker", "ai", 0.95, f"File metadata says it was made or edited with a generative tool ({m.decode(errors='ignore')!r})."))
            return out
    if any(m in low for m in C2PA_MARKERS):
        hint = next((h for h in C2PA_AI_HINTS if h in low), None)
        if hint:
            out.append(Finding("c2pa_ai", "ai", 0.9, f"File carries C2PA content credentials that mention a generative tool ({hint.decode()!r}). Verify the manifest with a C2PA tool."))
        else:
            out.append(Finding("c2pa_present", "info", 0.0, "File carries C2PA content credentials (also written by real cameras). Verify the manifest with a C2PA tool."))
    for m in WEAK_AI_MARKERS:
        if m in low:
            out.append(Finding("ai_marker_weak", "ai", 0.4, f"File contains a string associated with generative tools ({m.decode(errors='ignore')!r}); could be coincidence."))
            break
    return out


def _png_prompt_findings(path: Path) -> list[Finding]:
    try:
        with Image.open(path) as im:
            info = {str(k).lower(): str(v) for k, v in im.info.items()}
    except Exception:
        return []
    for k, v in info.items():
        if k in PNG_PROMPT_KEYS and re.search(r"steps:|sampler|cfg scale|negative prompt|\"class_type\"|seed", v, re.I):
            return [Finding("png_prompt", "ai", 0.97, "PNG text chunk contains generation parameters (prompt/sampler/seed), typical of Stable Diffusion / ComfyUI output.")]
    return []


def _pdf_checks(path: Path, raw: bytes) -> list[Finding]:
    from pypdf import PdfReader

    out: list[Finding] = []
    eofs = len(re.findall(rb"%%EOF", raw))
    if eofs > 1:
        out.append(Finding("incremental_update", "edit", 0.5, f"PDF was saved with {eofs} revisions (incremental updates); earlier versions may still be inside the file."))
    if b"/ByteRange" in raw:
        out.append(Finding("has_signature", "info", 0.0, "PDF contains a digital signature field. Validate it against the signer's trust chain; this tool does not."))
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        return out + [Finding("pdf_unreadable", "edit", 0.4, f"PDF could not be parsed: {e}")]

    info = reader.metadata or {}
    producer, creator = str(info.get("/Producer", "") or ""), str(info.get("/Creator", "") or "")
    blob = f"{producer} {creator}".lower()
    if any(t in blob for t in PDF_EDIT_TOOLS):
        out.append(Finding("pdf_edit_tool", "edit", 0.25, f"PDF producer/creator names an editing tool ({producer or creator!r})."))
    if any(t in blob for t in EDITOR_MARKERS):
        out.append(Finding("image_editor_in_pdf", "edit", 0.4, f"PDF producer/creator names an image editor ({producer or creator!r})."))
    created, modified = _parse_pdf_date(info.get("/CreationDate")), _parse_pdf_date(info.get("/ModDate"))
    if created and modified and (modified - created).total_seconds() > 120:
        out.append(Finding("modified_after_creation", "edit", 0.3, f"PDF was modified {(modified - created).total_seconds() / 3600:.1f} h after it was created."))

    for pno, page in enumerate(reader.pages[:5], start=1):
        try:
            fonts = page["/Resources"].get("/Font", {}) if "/Resources" in page else {}
            bases: dict[str, set[str]] = {}
            for f in fonts.values():
                m = re.match(r"/?([A-Z]{6})\+(.+)", str(f.get_object().get("/BaseFont", "")))
                if m:
                    bases.setdefault(m.group(2), set()).add(m.group(1))
            for base, tags in bases.items():
                if len(tags) > 1:
                    out.append(Finding("font_subset_mix", "edit", 0.35, f"Page {pno}: font {base} appears with {len(tags)} different subsets, which can mean re-typed text."))
        except Exception:
            pass
    return out


def _exif_checks(path: Path) -> list[Finding]:
    out: list[Finding] = []
    try:
        with Image.open(path) as im:
            im.load()
            exif, size = im.getexif(), im.size
    except Exception as e:
        return [Finding("image_unreadable", "edit", 0.4, f"Image could not be opened: {e}")]
    software = str(exif.get(305, "") or "")
    if software:
        if any(t in software.lower() for t in EDITOR_MARKERS):
            out.append(Finding("image_editor_exif", "edit", 0.4, f"EXIF Software tag names an image editor: {software!r}."))
        else:
            out.append(Finding("exif_software", "info", 0.0, f"EXIF Software tag: {software!r}."))
    dt = exif.get(306)
    try:
        dt_orig = exif.get_ifd(0x8769).get(36867)
    except Exception:
        dt_orig = None
    if dt and dt_orig and str(dt) != str(dt_orig):
        out.append(Finding("exif_time_mismatch", "edit", 0.15, "EXIF modification time differs from capture time."))
    if not exif:
        out.append(Finding("no_exif", "info", 0.0, "No EXIF metadata (normal for messaging apps, screenshots and scanners; also what AI generators produce)."))
        w, h = size
        if min(w, h) >= 512 and w % 64 == 0 and h % 64 == 0:
            out.append(Finding("generator_dimensions", "ai", 0.1, f"Image is {w}x{h}, both multiples of 64, which is a typical generator output size (very weak signal)."))
    if min(size) < 600:
        out.append(Finding("low_resolution", "info", 0.0, f"Image is only {size[0]}x{size[1]} px; detectors are less reliable at low resolution."))
    return out
