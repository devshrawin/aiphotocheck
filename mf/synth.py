"""Synthetic marksheet generator for harness testing.

IMPORTANT LIMITS
  * Every sample is clearly marked SYNTHETIC. These are NOT real boards' layouts.
  * The 'forged' edits here are classical/proxy edits (white-out + retype, copy-move, inpaint + retype).
    The 'inpaint_sim' edit only imitates a seamless edit; it is NOT a real generative-AI edit.
    For a meaningful evaluation, add REAL AI-edited samples (from your own genuine marksheets) as
        <data>/forged/ai_inpainted/*.jpg   and list them in labels.csv with edit_type=ai_inpainted.
  * Real marksheets must only be used with the proper authority and data-protection approvals.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

DPI = 150
S = DPI / 72.0
PAGE_W, PAGE_H = A4
SUBJECTS = ["English", "Malayalam", "Mathematics", "Physics", "Chemistry", "Social Science"]
FIRST = ["Anjali", "Arun", "Devika", "Fahad", "Gopika", "Hari", "Irfan", "Jyothi", "Kiran", "Lakshmi",
         "Manu", "Nandana", "Midhun", "Reshma", "Sandeep", "Thara", "Unni", "Vishnu", "Yamini", "Zain"]
LAST = ["Nair", "Menon", "Pillai", "Thomas", "Kurian", "Mohammed", "Varghese", "Das", "Krishnan", "Joseph"]

FIELD_X, NAME_Y, ROLL_Y, DOB_Y = 60, 740, 718, 696
COL_SUB, COL_MAX, COL_OBT = 60, 330, 430
ROW0_Y, ROW_STEP = 635, 24
TOTAL_Y, PCT_Y, RESULT_Y = 480, 456, 432
FONT_PT = 12


@dataclass
class Sheet:
    sid: str
    name: str
    roll: str
    dob: str
    marks: list[int]

    @property
    def total(self):
        return sum(self.marks)

    @property
    def pct(self):
        return round(100.0 * self.total / (100 * len(self.marks)), 2)

    def registry(self):
        return {"name": self.name, "roll": self.roll, "dob": self.dob}


def random_sheet(rng, sid: str) -> Sheet:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    return Sheet(sid, name, str(rng.integers(100000, 999999)),
                 f"{rng.integers(1, 28):02d}-{rng.integers(1, 12):02d}-{rng.integers(2006, 2009)}",
                 [int(x) for x in rng.integers(38, 98, size=len(SUBJECTS))])


def render_pdf(sh: Sheet, path: Path):
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"Sample marksheet {sh.sid}")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(120, 800, "STATE BOARD OF SECONDARY EXAMINATIONS")
    c.setFont("Helvetica", 9)
    c.drawString(150, 784, "SYNTHETIC SAMPLE - FOR SOFTWARE TESTING ONLY")
    c.setFont("Helvetica", FONT_PT)
    c.drawString(FIELD_X, NAME_Y, f"Name of Candidate: {sh.name}")
    c.drawString(FIELD_X, ROLL_Y, f"Roll No: {sh.roll}")
    c.drawString(FIELD_X, DOB_Y, f"Date of Birth: {sh.dob}")
    c.setFont("Helvetica-Bold", FONT_PT)
    c.drawString(COL_SUB, 660, "Subject"); c.drawString(COL_MAX, 660, "Max"); c.drawString(COL_OBT, 660, "Obtained")
    c.setFont("Helvetica", FONT_PT)
    for i, (sub, m) in enumerate(zip(SUBJECTS, sh.marks)):
        y = ROW0_Y - i * ROW_STEP
        c.drawString(COL_SUB, y, sub); c.drawString(COL_MAX, y, "100"); c.drawString(COL_OBT, y, str(m))
    c.setFont("Helvetica-Bold", FONT_PT)
    c.drawString(COL_SUB, TOTAL_Y, "GRAND TOTAL"); c.drawString(COL_MAX, TOTAL_Y, str(100 * len(SUBJECTS))); c.drawString(COL_OBT, TOTAL_Y, str(sh.total))
    c.drawString(COL_SUB, PCT_Y, "PERCENTAGE"); c.drawString(COL_OBT, PCT_Y, f"{sh.pct:.2f}")
    c.drawString(COL_SUB, RESULT_Y, "RESULT"); c.drawString(COL_OBT, RESULT_Y, "PASS")
    c.line(55, 672, 540, 672)
    c.save()


def rasterize(pdf: Path, out_png: Path) -> Image.Image:
    prefix = out_png.with_suffix("")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(pdf), str(prefix)], check=True)
    return Image.open(out_png).convert("RGB")


# ---------- cell geometry (pixels) ----------
def _rect(x_pt, y_pt, text, bold=False):
    f = "Helvetica-Bold" if bold else "Helvetica"
    w = stringWidth(text, f, FONT_PT)
    return (int((x_pt - 3) * S), int((PAGE_H - (y_pt + 10)) * S), int((x_pt + w + 5) * S), int((PAGE_H - (y_pt - 3)) * S))


def cell_rects(sh: Sheet) -> dict:
    return {
        "obt": [_rect(COL_OBT, ROW0_Y - i * ROW_STEP, str(m)) for i, m in enumerate(sh.marks)],
        "total": _rect(COL_OBT, TOTAL_Y, str(sh.total), bold=True),
        "pct": _rect(COL_OBT, PCT_Y, f"{sh.pct:.2f}", bold=True),
        "name": _rect(FIELD_X + stringWidth("Name of Candidate: ", "Helvetica", FONT_PT), NAME_Y, sh.name),
    }


_FONT_CACHE: dict = {}


def _font(bold: bool, px: int):
    import matplotlib
    key = (bold, px)
    if key not in _FONT_CACHE:
        fn = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        _FONT_CACHE[key] = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / fn), px)
    return _FONT_CACHE[key]


def retype(img: Image.Image, rect, text: str, soft: bool, bold=False, rng=None) -> Image.Image:
    """Replace the contents of `rect` with `text` in a different font. soft=True imitates a seamless edit."""
    l, t, r, b = rect
    arr = np.asarray(img).copy()
    if soft:
        mask = np.zeros(arr.shape[:2], np.uint8)
        mask[t:b, l:r] = 255
        arr = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
        pil = Image.fromarray(arr)
    else:
        pil = Image.fromarray(arr)
        ImageDraw.Draw(pil).rectangle([l, t, r, b], fill=(255, 255, 255))
    d = ImageDraw.Draw(pil)
    px = int(FONT_PT * S * 0.98)
    d.text((l + int(3 * S), b - int(3 * S)), text, font=_font(bold, px), fill=(25, 25, 25) if soft else (0, 0, 0), anchor="ls")
    if soft:  # blend the patch: soften edges and add matching grain
        reg = np.asarray(pil)[t - 4:b + 4, l - 4:r + 4].astype(np.float32)
        reg = cv2.GaussianBlur(reg, (0, 0), 0.6)
        reg += (rng or np.random.default_rng(0)).normal(0, 2.0, reg.shape)
        out = np.asarray(pil).copy()
        out[t - 4:b + 4, l - 4:r + 4] = np.clip(reg, 0, 255).astype(np.uint8)
        pil = Image.fromarray(out)
    return pil


def forge(img: Image.Image, sh: Sheet, kind: str, rng) -> Image.Image:
    rects = cell_rects(sh)
    k = int(rng.integers(0, len(SUBJECTS)))
    new = min(100, sh.marks[k] + int(rng.integers(8, 20)))
    if new == sh.marks[k]:
        new -= 12
    delta = new - sh.marks[k]
    if kind in ("edit_inconsistent",):
        return retype(img, rects["obt"][k], str(new), soft=False, rng=rng)
    if kind in ("edit_consistent", "inpaint_sim"):
        soft = kind == "inpaint_sim"
        tot = sh.total + delta
        pct = round(100.0 * tot / (100 * len(SUBJECTS)), 2)
        out = retype(img, rects["obt"][k], str(new), soft, rng=rng)
        out = retype(out, rects["total"], str(tot), soft, bold=True, rng=rng)
        return retype(out, rects["pct"], f"{pct:.2f}", soft, bold=True, rng=rng)
    if kind == "copy_move":
        j = next(i for i in range(len(SUBJECTS)) if sh.marks[i] != sh.marks[k])
        l, t, r, b = rects["obt"][j]
        patch = img.crop((l, t, r, b))
        out = img.copy()
        L, T, R, B = rects["obt"][k]
        out.paste(Image.new("RGB", (R - L, B - T), (255, 255, 255)), (L, T))
        out.paste(patch, (L, T))
        return out
    if kind == "name_swap":
        other = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        return retype(img, rects["name"], other, soft=True, rng=rng)
    raise ValueError(kind)


# ---------- capture simulation ----------
def scan_like(img: Image.Image, rng) -> tuple[Image.Image, int]:
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-0.6, 0.6)), 1.0)
    a = cv2.warpAffine(a, M, (w, h), borderValue=(255, 255, 255))
    a = cv2.GaussianBlur(a, (0, 0), float(rng.uniform(0.5, 0.9)))
    a = a * float(rng.uniform(0.94, 1.0)) + rng.normal(0, rng.uniform(2, 4), a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), int(rng.integers(75, 91))


def phone_like(img: Image.Image, rng) -> tuple[Image.Image, int]:
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    j = lambda: float(rng.uniform(-0.02, 0.02))  # noqa: E731
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * (0.01 + abs(j())), h * (0.005 + abs(j()))], [w * (0.99 - abs(j())), h * (0.01 + abs(j()))],
                      [w * (0.99 - abs(j())), h * (0.99 - abs(j()))], [w * (0.01 + abs(j())), h * (0.995 - abs(j()))]])
    a = cv2.warpPerspective(a, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(215, 212, 205))
    grad = np.linspace(rng.uniform(0.82, 0.95), rng.uniform(0.97, 1.05), w)[None, :, None]
    a = a * grad
    a = cv2.GaussianBlur(a, (0, 0), float(rng.uniform(0.8, 1.3)))
    a += rng.normal(0, rng.uniform(3, 6), a.shape)
    img2 = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img2 = img2.resize((1400, int(1400 * h / w)), Image.LANCZOS)
    return img2, int(rng.integers(55, 81))


def save_capture(img: Image.Image, q: int, path_noext: Path, as_pdf: bool) -> Path:
    if as_pdf:
        p = path_noext.with_suffix(".pdf")
        img.save(p, "PDF", resolution=DPI, quality=q)
    else:
        p = path_noext.with_suffix(".jpg")
        img.save(p, "JPEG", quality=q)
    return p


def edit_pdf_text(pdf_in: Path, pdf_out: Path, x_pt: float, y_pt: float, new_text: str) -> bool:
    """Replace the text drawn at (x, y) in a reportlab PDF's content stream (a no-trace text-layer edit)."""
    import pikepdf

    with pikepdf.open(pdf_in) as pdf:
        page = pdf.pages[0]
        data = page.Contents.read_bytes() if not isinstance(page.Contents, pikepdf.Array) else b"".join(c.read_bytes() for c in page.Contents)
        pat = re.compile(rb"(1 0 0 1 %s %s Tm \()[^)]*(\) Tj)" % (re.escape(_num(x_pt)), re.escape(_num(y_pt))))
        new, n = pat.subn(lambda m: m.group(1) + new_text.encode() + m.group(2), data, count=1)
        if n == 0:
            return False
        page.Contents = pdf.make_stream(new)
        pdf.save(pdf_out)
    return True


def _num(v: float) -> bytes:
    return (str(int(v)) if float(v).is_integer() else ("%f" % v).rstrip("0").rstrip(".")).encode()


FORGERY_KINDS = ["edit_inconsistent", "edit_consistent", "copy_move", "inpaint_sim", "name_swap"]


def make_dataset(out: Path, n: int = 12, seed: int = 7) -> Path:
    out = Path(out)
    rng = np.random.default_rng(seed)
    for sub in ("genuine", "forged", "_tmp"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, registry = [], {}
    for i in range(n):
        sid = f"S{i + 1:03d}"
        sh = random_sheet(rng, sid)
        registry[sid] = sh.registry()
        tmp_pdf, tmp_png = out / "_tmp" / f"{sid}.pdf", out / "_tmp" / f"{sid}.png"
        render_pdf(sh, tmp_pdf)
        clean = rasterize(tmp_pdf, tmp_png)

        # genuine: original PDF, scan (jpg or pdf), phone photo
        g = out / "genuine"
        (g / f"{sid}__pdf_clean.pdf").write_bytes(tmp_pdf.read_bytes())
        rows.append((f"genuine/{sid}__pdf_clean.pdf", "genuine", "none", "pdf_clean", sid))
        im, q = scan_like(clean, rng)
        p = save_capture(im, q, g / f"{sid}__scan", as_pdf=bool(i % 3 == 0))
        rows.append((f"genuine/{p.name}", "genuine", "none", "scan", sid))
        im, q = phone_like(clean, rng)
        p = save_capture(im, q, g / f"{sid}__phone", as_pdf=False)
        rows.append((f"genuine/{p.name}", "genuine", "none", "phone", sid))

        # forged: each kind through a scan or phone capture
        for j, kind in enumerate(FORGERY_KINDS):
            (out / "forged" / kind).mkdir(parents=True, exist_ok=True)
            forged = forge(clean, sh, kind, rng)
            carrier = "scan" if (i + j) % 2 == 0 else "phone"
            im, q = (scan_like if carrier == "scan" else phone_like)(forged, rng)
            p = save_capture(im, q, out / "forged" / kind / f"{sid}__{carrier}_{kind}", as_pdf=bool(carrier == "scan" and (i + j) % 4 == 0))
            rows.append((f"forged/{kind}/{p.name}", "forged", kind, carrier, sid))

        # forged: text-layer edit inside the PDF itself (leaves no editing-tool trace)
        (out / "forged" / "pdf_text_edit").mkdir(parents=True, exist_ok=True)
        k = int(rng.integers(0, len(SUBJECTS)))
        dst = out / "forged" / "pdf_text_edit" / f"{sid}__pdf_text_edit.pdf"
        if edit_pdf_text(tmp_pdf, dst, COL_OBT, ROW0_Y - k * ROW_STEP, str(min(100, sh.marks[k] + 11))):
            rows.append((f"forged/pdf_text_edit/{dst.name}", "forged", "pdf_text_edit", "pdf_clean", sid))

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "edit_type", "carrier", "sample_id"])
        w.writerows(rows)
    (out / "registry.json").write_text(json.dumps(registry, indent=2))
    for t in (out / "_tmp").glob("*"):
        t.unlink()
    (out / "_tmp").rmdir()
    return out / "labels.csv"
