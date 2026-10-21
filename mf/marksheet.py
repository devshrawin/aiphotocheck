"""Content checks: OCR the marksheet, verify its own arithmetic, cross-check against registry data.

The parsing patterns live in config.json ("marksheet"). Real boards each have their own layout, so
production use needs one template per board/year. The defaults match the synthetic samples.
"""
from __future__ import annotations

import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image, ImageOps

from .rules import Finding


def ocr_image(path: Path, variant: int = 0) -> str:
    """variant 0: autocontrast at ~1800 px. variant 1: larger, blurred and Otsu-binarised.
    Two different preprocessings let us tell a real inconsistency from an OCR misread."""
    import cv2
    import numpy as np
    import pytesseract

    im = Image.open(path).convert("L")
    target = 1800 if variant == 0 else 2600
    if max(im.size) != target:
        s = target / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    if variant == 0:
        im = ImageOps.autocontrast(im)
    else:
        a = cv2.GaussianBlur(np.asarray(im), (0, 0), 1.0)
        _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im = Image.fromarray(a)
    return pytesseract.image_to_string(im, config="--psm 6")


def pdf_text_layer(path: Path, max_pages: int = 3) -> str:
    """Text layer with table rows kept on one line (plain extract_text() puts every cell on its own line)."""
    import subprocess

    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", "-l", str(max_pages), str(path), "-"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout
    from pypdf import PdfReader

    try:
        rd = PdfReader(str(path))
        return "\n".join((p.extract_text(extraction_mode="layout") or "") for p in rd.pages[:max_pages])
    except Exception:
        return ""


def get_texts(path: Path, kind: str, images: list[tuple[Path, str]]) -> tuple[list[str], str]:
    """Return ([text variants], source) with source in {'text_layer', 'ocr', 'none'}."""
    if kind == "pdf":
        t = pdf_text_layer(path)
        if len(t.strip()) >= 40:
            return [t], "text_layer"
    if shutil.which("tesseract") is None or not images:
        return [], "none"
    try:
        return ["\n".join(ocr_image(p, v) for p, _ in images[:2]) for v in (0, 1)], "ocr"
    except Exception:
        return [], "none"


def parse(text: str, pats: dict) -> dict:
    subjects = []
    for line in text.splitlines():
        m = re.match(pats["subject_line"], line)
        if m and "total" not in m.group("subject").lower():
            subjects.append((m.group("subject").strip(), int(m.group("max")), int(m.group("obt"))))
    gt = re.search(pats["grand_total"], text)
    pct = re.search(pats["percentage"], text)

    def field(key):
        m = re.search(pats[key], text)
        return m.group("v").strip() if m else None

    return {
        "subjects": subjects,
        "total_max": int(gt.group("max")) if gt else None,
        "total_obt": int(gt.group("obt")) if gt else None,
        "percentage": float(pct.group("pct")) if pct else None,
        "name": field("name"), "roll": field("roll"), "dob": field("dob"),
    }


def arithmetic_findings(p: dict) -> list[Finding]:
    out: list[Finding] = []
    subs = p["subjects"]
    if len(subs) < 3 or p["total_obt"] is None:
        return [Finding("marks_not_parsed", 0.0, "Could not read enough subject rows/totals to check arithmetic (template mismatch or poor OCR).")]
    for name, mx, obt in subs:
        if obt > mx:
            out.append(Finding("marks_exceed_max", 0.6, f"{name}: obtained {obt} exceeds maximum {mx}."))
    s_obt, s_max = sum(o for _, _, o in subs), sum(m for _, m, _ in subs)
    if s_obt != p["total_obt"]:
        out.append(Finding("total_mismatch", 0.6, f"Subject marks add up to {s_obt} but the marksheet states a total of {p['total_obt']}."))
    if p["total_max"] is not None and s_max != p["total_max"]:
        out.append(Finding("max_total_mismatch", 0.4, f"Subject maximums add up to {s_max} but the marksheet states {p['total_max']}."))
    if p["percentage"] is not None and p["total_max"]:
        exp = 100.0 * p["total_obt"] / p["total_max"]
        if abs(exp - p["percentage"]) > 0.15:
            out.append(Finding("percentage_mismatch", 0.5, f"Stated percentage {p['percentage']} does not match total/maximum ({exp:.2f})."))
    return out


def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def registry_findings(p: dict, expected: dict | None) -> list[Finding]:
    if not expected:
        return []
    out: list[Finding] = []
    if p["name"] and expected.get("name"):
        ratio = SequenceMatcher(None, _norm_name(p["name"]), _norm_name(expected["name"])).ratio()
        if ratio < 0.85:
            out.append(Finding("registry_name_mismatch", 0.5, f"Name on document ({p['name']!r}) does not match the registry record."))
    if p["roll"] and expected.get("roll") and _digits(p["roll"]) != _digits(expected["roll"]):
        out.append(Finding("registry_roll_mismatch", 0.5, "Roll number on document does not match the registry record."))
    if p["dob"] and expected.get("dob") and _digits(p["dob"]) != _digits(expected["dob"]):
        out.append(Finding("registry_dob_mismatch", 0.5, "Date of birth on document does not match the registry record."))
    return out


def _merge(finding_sets: list[list[Finding]]) -> list[Finding]:
    """Keep a finding at full weight only if every OCR pass produced it; otherwise heavily discount it."""
    if len(finding_sets) == 1:
        return finding_sets[0]
    confirmed = set.intersection(*[{f.id for f in fs} for fs in finding_sets])
    out, seen = [], set()
    for fs in finding_sets:
        for f in fs:
            if f.id in seen:
                continue
            seen.add(f.id)
            if f.id in confirmed or f.weight == 0:
                out.append(f)
            else:
                out.append(Finding(f.id, round(f.weight * 0.25, 3),
                                   f.message + " (seen in only one of two OCR passes, so it may be a misreading)"))
    return out


def run_checks(path: Path, kind: str, images, cfg: dict, expected: dict | None):
    texts, source = get_texts(path, kind, images)
    if not any(t.strip() for t in texts):
        return [Finding("no_text", 0.0, "No text could be read from the document.")], {"source": source}
    parsed = [parse(t, cfg["marksheet"]) for t in texts]
    findings = _merge([arithmetic_findings(p) + registry_findings(p, expected) for p in parsed])
    best = max(parsed, key=lambda p: len(p["subjects"]))
    return findings, {"source": source, "parsed": best}
