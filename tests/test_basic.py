import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mf import marksheet, pipeline, rules, synth
from mf.detectors.classical import ClassicalDetector

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rng = np.random.default_rng(3)
        self.sheet = synth.random_sheet(self.rng, "T1")
        self.pdf = self.tmp / "a.pdf"
        synth.render_pdf(self.sheet, self.pdf)


class RulesTests(Base):
    def test_sniff(self):
        self.assertEqual(rules.sniff(self.pdf), "pdf")

    def test_incremental_update_detected(self):
        from pypdf import PdfWriter

        out = self.tmp / "b.pdf"
        shutil.copy(self.pdf, out)
        w = PdfWriter(fileobj=str(out), clone_from=str(out), incremental=True)
        w.add_metadata({"/Producer": "iLovePDF"})
        w.write(str(out))
        ids = {f.id for f in rules.file_checks(out, "pdf")}
        self.assertIn("incremental_update", ids)
        self.assertIn("pdf_edit_tool", ids)

    def test_clean_pdf_has_no_findings(self):
        self.assertEqual([f for f in rules.file_checks(self.pdf, "pdf") if f.weight > 0], [])

    def test_ai_marker(self):
        p = self.tmp / "x.jpg"
        from PIL import Image
        Image.new("RGB", (50, 50)).save(p)
        p.write_bytes(p.read_bytes() + b"trainedAlgorithmicMedia")
        self.assertIn("ai_marker", {f.id for f in rules.file_checks(p, "jpeg")})


class ContentTests(Base):
    def test_text_layer_parses_and_edit_is_caught(self):
        t, src = marksheet.get_texts(self.pdf, "pdf", [])
        p = marksheet.parse(t[0], CFG["marksheet"])
        self.assertEqual(src, "text_layer")
        self.assertEqual(len(p["subjects"]), 6)
        self.assertEqual(marksheet.arithmetic_findings(p), [])
        edited = self.tmp / "e.pdf"
        self.assertTrue(synth.edit_pdf_text(self.pdf, edited, synth.COL_OBT, synth.ROW0_Y, "99" if self.sheet.marks[0] != 99 else "98"))
        t, _ = marksheet.get_texts(edited, "pdf", [])
        ids = {f.id for f in marksheet.arithmetic_findings(marksheet.parse(t[0], CFG["marksheet"]))}
        self.assertIn("total_mismatch", ids)

    def test_registry_mismatch(self):
        p = {"name": "Someone Else", "roll": self.sheet.roll, "dob": self.sheet.dob}
        ids = {f.id for f in marksheet.registry_findings(p, self.sheet.registry())}
        self.assertEqual(ids, {"registry_name_mismatch"})


class DetectorAndPipelineTests(Base):
    def test_classical_detector_makes_heatmap(self):
        png = self.tmp / "a.png"
        img = synth.rasterize(self.pdf, png)
        r = ClassicalDetector().score(png, self.tmp)
        self.assertTrue(0.0 <= r.score <= 1.0)
        self.assertTrue(Path(r.heatmap_path).exists())

    def test_citizen_message_never_reveals_score_or_ai(self):
        for band, msg in pipeline.CITIZEN_MESSAGES.items():
            low = msg.lower()
            for bad in ("ai", "altered", "forged", "fake", "%", "score"):
                self.assertNotIn(bad if bad != "ai" else " ai ", f" {low} " if bad == "ai" else low, msg=band)

    def test_noisy_or(self):
        self.assertAlmostEqual(pipeline.noisy_or([0.5, 0.5]), 0.75)
        self.assertEqual(pipeline.noisy_or([]), 0.0)

    def test_analyze_pdf_end_to_end(self):
        rep = pipeline.analyze(self.pdf, CFG, [ClassicalDetector()], self.tmp / "w", self.sheet.registry())
        self.assertEqual(rep["band"], "low")
        self.assertNotIn("score", rep["citizen_message"].lower())


if __name__ == "__main__":
    unittest.main()
