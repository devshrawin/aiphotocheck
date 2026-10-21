"""Rules layer: cheap, explainable checks on file structure and metadata.

Every finding has a weight in [0, 1]. Weights are combined with a noisy-OR
(see pipeline.combine). They are heuristics, NOT probabilities.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Strings that, if present anywhere in the file bytes, point at generative tools / provenance data.
AI_MARKERS = [
    b"trainedalgorithmicmedia", b"stable diffusion", b"midjourney", b"dall-e", b"dall\xc2\xb7e",
    b"firefly", b"gpt-image", b"ideogram", b"imagen", b"nano banana", b"c2pa", b"jumbf",
]
EDITOR_MARKERS = [
    "photoshop", "gimp", "pixlr", "canva", "snapseed", "paint.net", "illustrator",
    "affinity", "lightroom", "photopea", "picsart",
]
PDF_EDIT_TOOLS = [
    "ilovepdf", "smallpdf", "sejda", "pdf-xchange", "foxit", "nitro", "wondershare",
    "pdfescape", "sodapdf", "pdfelement", "acrobat", "pdffiller", "pdf24",
]


@dataclass
class Finding:
    id: str
    weight: float
    message: str

    def to_dict(self):
        return asdict(self)


def sniff(path: Path) -> str:
    head = Path(path).read_bytes()[:16]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def _parse_pdf_date(s) -> datetime | None:
    if not s:
        return None
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", str(s))
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
    ext_kind = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "pdf": "pdf", "tif": "tiff",
                "tiff": "tiff", "webp": "webp"}.get(ext)
    if kind == "unknown":
        out.append(Finding("unknown_type", 0.3, "File type could not be identified from its contents."))
        return out
    if ext_kind and ext_kind != kind:
        out.append(Finding("extension_mismatch", 0.3, f"File extension .{ext} does not match actual content ({kind})."))

    raw = path.read_bytes()
    low = raw.lower()
    for marker in AI_MARKERS:
        if marker in low:
            weight = 0.9 if marker in (b"trainedalgorithmicmedia", b"gpt-image", b"midjourney", b"stable diffusion") else 0.5
            out.append(Finding("ai_marker", weight, f"File contains a generative-AI / provenance marker: {marker.decode(errors='ignore')!r}. "
                                                   "Provenance data is informative but not proof; verify manifests with a C2PA tool."))
            break

    if kind == "pdf":
        out += _pdf_checks(path, raw)
    else:
        out += _exif_checks(path)
    return out


def _pdf_checks(path: Path, raw: bytes) -> list[Finding]:
    from pypdf import PdfReader

    out: list[Finding] = []
    eofs = len(re.findall(rb"%%EOF", raw))
    if eofs > 1:
        out.append(Finding("incremental_update", 0.5,
                           f"PDF was saved with {eofs} revisions (incremental updates). Earlier versions may still be inside the file."))
    if b"/ByteRange" in raw:
        out.append(Finding("has_signature", 0.0,
                           "PDF contains a digital signature field. Validate it with the issuer's trust chain; this tool does not."))
    try:
        reader = PdfReader(str(path))
    except Exception as e:  # corrupt / encrypted
        out.append(Finding("pdf_unreadable", 0.4, f"PDF could not be parsed: {e}"))
        return out

    info = reader.metadata or {}
    producer = str(info.get("/Producer", "") or "")
    creator = str(info.get("/Creator", "") or "")
    tool_blob = f"{producer} {creator}".lower()
    for t in PDF_EDIT_TOOLS:
        if t in tool_blob:
            out.append(Finding("pdf_edit_tool", 0.25, f"PDF producer/creator names an editing tool ({producer or creator!r})."))
            break
    for t in EDITOR_MARKERS:
        if t in tool_blob:
            out.append(Finding("image_editor_in_pdf", 0.4, f"PDF producer/creator names an image editor ({producer or creator!r})."))
            break
    created, modified = _parse_pdf_date(info.get("/CreationDate")), _parse_pdf_date(info.get("/ModDate"))
    if created and modified and (modified - created).total_seconds() > 120:
        out.append(Finding("modified_after_creation", 0.3,
                           f"PDF was modified {(modified - created).total_seconds() / 3600:.1f} h after it was created."))

    for pno, page in enumerate(reader.pages[:5], start=1):
        # Same base font with several different subset tags on one page -> text was probably re-typed.
        try:
            fonts = page["/Resources"].get("/Font", {}) if "/Resources" in page else {}
            bases = {}
            for f in fonts.values():
                name = str(f.get_object().get("/BaseFont", ""))
                m = re.match(r"/?([A-Z]{6})\+(.+)", name)
                if m:
                    bases.setdefault(m.group(2), set()).add(m.group(1))
            for base, tags in bases.items():
                if len(tags) > 1:
                    out.append(Finding("font_subset_mix", 0.35,
                                       f"Page {pno}: font {base} appears with {len(tags)} different subsets, which can indicate re-typed text."))
        except Exception:
            pass
        try:
            n_img = len(page.images)
        except Exception:
            n_img = 0
        if n_img > 1:
            out.append(Finding("multiple_images_on_page", 0.3,
                               f"Page {pno} contains {n_img} separate images (a scan with a pasted patch looks like this, but so do legitimate stamps/photos)."))
    return out


def _exif_checks(path: Path) -> list[Finding]:
    out: list[Finding] = []
    try:
        with Image.open(path) as im:
            im.load()
            exif, size = im.getexif(), im.size
    except Exception as e:
        return [Finding("image_unreadable", 0.4, f"Image could not be opened: {e}")]
    software = str(exif.get(305, "") or "")  # 305 = Software
    dt, dt_orig = exif.get(306), None
    try:
        dt_orig = exif.get_ifd(0x8769).get(36867)
    except Exception:
        pass
    if software:
        for t in EDITOR_MARKERS:
            if t in software.lower():
                out.append(Finding("image_editor_exif", 0.4, f"EXIF Software tag names an image editor: {software!r}."))
                break
    if dt and dt_orig and str(dt) != str(dt_orig):
        out.append(Finding("exif_time_mismatch", 0.15, "EXIF modification time differs from capture time."))
    if not exif:
        out.append(Finding("no_exif", 0.03, "No EXIF metadata (common for messaging apps and scanners, so a very weak signal)."))
    if min(size) < 600:
        out.append(Finding("low_resolution", 0.0, f"Image is only {size[0]}x{size[1]} px; forensic detectors are less reliable at low resolution."))
    return out


def image_checks(path: Path) -> list[Finding]:
    """Checks on an image file that was embedded in a PDF or uploaded directly."""
    return _exif_checks(Path(path)) if sniff(Path(path)) in ("jpeg", "png", "tiff", "webp") else []


def extract_images(path: Path, kind: str, workdir: Path, dpi: int = 150, max_pages: int = 5) -> list[tuple[Path, str]]:
    """Return [(image_path, origin)] where origin is 'upload', 'embedded' or 'raster'."""
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if kind != "pdf":
        return [(path, "upload")] if kind in ("jpeg", "png", "tiff", "webp") else []
    from pypdf import PdfReader

    imgs: list[tuple[Path, str]] = []
    try:
        reader = PdfReader(str(path))
        n_pages = min(len(reader.pages), max_pages)
        for pno in range(n_pages):
            page_imgs = []
            try:
                for k, im in enumerate(reader.pages[pno].images):
                    data = im.data
                    ext = "jpg" if data[:3] == b"\xff\xd8\xff" else "png"
                    p = workdir / f"{path.stem}_p{pno + 1}_img{k + 1}.{ext}"
                    if ext == "jpg":
                        p.write_bytes(data)
                    else:
                        im.image.convert("RGB").save(p)
                    page_imgs.append((p, "embedded"))
            except Exception:
                page_imgs = []
            if page_imgs:
                imgs += page_imgs
            else:
                prefix = workdir / f"{path.stem}_p{pno + 1}_raster"
                subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(pno + 1), "-l", str(pno + 1), "-png", "-singlefile",
                                str(path), str(prefix)], check=False, capture_output=True)
                if prefix.with_suffix(".png").exists():
                    imgs.append((prefix.with_suffix(".png"), "raster"))
    except Exception:
        pass
    return imgs
