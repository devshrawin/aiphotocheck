"""File-type sniffing and image extraction, so any attachment can be fed to the image detectors."""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

IMAGE_KINDS = ("jpeg", "png", "tiff", "webp", "gif", "bmp")
OFFICE_MEDIA_DIRS = {"docx": "word/media/", "pptx": "ppt/media/", "xlsx": "xl/media/"}
MAX_IMAGES = 20


def sniff(path: Path) -> str:
    """File type from content, never from the extension."""
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
    if head[:4] == b"GIF8":
        return "gif"
    if head[:2] == b"BM":
        return "bmp"
    if head[:4] == b"PK\x03\x04":
        try:
            names = zipfile.ZipFile(path).namelist()
        except zipfile.BadZipFile:
            return "unknown"
        for kind, media in OFFICE_MEDIA_DIRS.items():
            if any(n.startswith(media.rsplit("/", 2)[0] + "/") for n in names):
                return kind
    return "unknown"


def extract_images(path: Path, kind: str, workdir: Path, dpi: int = 150, max_pages: int = 5) -> list[tuple[Path, str]]:
    """Return [(image_path, origin)], origin in 'upload', 'embedded' or 'raster'."""
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if kind in IMAGE_KINDS:
        return [(path, "upload")]
    if kind in OFFICE_MEDIA_DIRS:
        return _office_images(path, kind, workdir)
    if kind == "pdf":
        return _pdf_images(path, workdir, dpi, max_pages)
    return []


def _office_images(path: Path, kind: str, workdir: Path) -> list[tuple[Path, str]]:
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith(OFFICE_MEDIA_DIRS[kind]) and n.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png", "webp", "tif", "tiff", "gif", "bmp"):
                p = workdir / f"{path.stem}_{Path(n).name}"
                p.write_bytes(z.read(n))
                out.append((p, "embedded"))
                if len(out) >= MAX_IMAGES:
                    break
    return out


def _pdf_images(path: Path, workdir: Path, dpi: int, max_pages: int) -> list[tuple[Path, str]]:
    from pypdf import PdfReader

    imgs: list[tuple[Path, str]] = []
    try:
        reader = PdfReader(str(path))
        for pno in range(min(len(reader.pages), max_pages)):
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
            else:  # vector/text page: render it so the detectors still see something
                prefix = workdir / f"{path.stem}_p{pno + 1}_raster"
                subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(pno + 1), "-l", str(pno + 1), "-png", "-singlefile",
                                str(path), str(prefix)], check=False, capture_output=True)
                if prefix.with_suffix(".png").exists():
                    imgs.append((prefix.with_suffix(".png"), "raster"))
    except Exception:
        pass
    return imgs[:MAX_IMAGES]
