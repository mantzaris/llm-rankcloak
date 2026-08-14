import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DESIGN = ROOT / "config" / "design.json"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class HumanStudyPackageTests(unittest.TestCase):
    def test_instrument_has_seven_complete_scales(self):
        instrument = json.loads((ROOT / "survey" / "instrument.json").read_text())
        scales = instrument["scales"]
        self.assertEqual(len(scales), 7)
        self.assertEqual(len({scale["scale_id"] for scale in scales}), 7)
        self.assertEqual(
            {scale["scale_id"] for scale in scales if scale["role"] == "co_primary"},
            {"overall_naturalness", "suspiciousness"},
        )
        for scale in scales:
            self.assertEqual(set(scale["anchors"]), {"1", "4", "7"})
            self.assertTrue(scale["prompt"])
            self.assertTrue(scale["definition"])

    def test_irb_materials_are_prominent_drafts_and_placeholders_are_inventoried(self):
        inventory = json.loads((ROOT / "config" / "placeholders.json").read_text())
        allowed = set(inventory["required_before_irb_submission"])
        observed = set()
        for path in ROOT.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            observed.update(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
        self.assertFalse(observed - allowed, "uninventoried placeholders: {}".format(observed - allowed))
        self.assertTrue(observed)
        for name in (
            "DRAFT_PROTOCOL.md", "CONSENT_INFORMATION_SHEET.md",
            "PARTICIPANT_INSTRUCTIONS.md", "DEBRIEF_TEXT.md",
        ):
            text = (ROOT / "irb" / name).read_text(encoding="utf-8").upper()
            self.assertIn("DRAFT", text)
            self.assertIn("NOT", text)
            self.assertIn("APPROV", text)

    def test_design_arithmetic(self):
        design = json.loads(DESIGN.read_text())
        self.assertEqual(len(design["conditions"]), 8)
        self.assertEqual(len(design["prompt_categories"]), 6)
        self.assertEqual(design["templates_per_category"], 3)
        self.assertEqual(design["stimuli_per_condition"], 72)
        self.assertEqual(
            len(design["conditions"]) * design["stimuli_per_condition"]
            * design["ratings_per_stimulus"],
            1728,
        )

    def test_end_to_end_synthetic_selection_schedule_and_analysis_fixture(self):
        design = json.loads(DESIGN.read_text())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            candidates = directory / "candidates.csv"
            randomization = directory / "randomization"
            ratings = directory / "ratings.csv"
            subprocess.run(
                [sys.executable, str(ROOT / "randomization" / "generate_synthetic_candidates.py"),
                 "--output", str(candidates)],
                cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, text=True,
            )
            candidate_rows = read_csv(candidates)
            self.assertEqual(len(candidate_rows), 1152)
            self.assertTrue(all(row["synthetic_fixture"] == "true" for row in candidate_rows))

            command = [
                sys.executable, str(ROOT / "randomization" / "build_blinded_assignments.py"),
                "--candidates", str(candidates), "--output-dir", str(randomization),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, text=True)
            first_hash = hashlib.sha256((randomization / "participant_schedule.csv").read_bytes()).hexdigest()
            subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, text=True)
            second_hash = hashlib.sha256((randomization / "participant_schedule.csv").read_bytes()).hexdigest()
            self.assertEqual(first_hash, second_hash)

            key = read_csv(randomization / "blind_key.csv")
            schedule = read_csv(randomization / "participant_schedule.csv")
            self.assertEqual(len(key), 576)
            self.assertEqual(len(schedule), 72 * 26)
            experimental = [row for row in schedule if row["item_type"] == "experimental_message"]
            checks = [row for row in schedule if row["item_type"] == "attention_check"]
            self.assertEqual(len(experimental), 1728)
            self.assertEqual(len(checks), 144)
            self.assertEqual(set(experimental[0]) & {"condition", "model_family", "payload_class"}, set())
            self.assertTrue(all(count == 3 for count in Counter(
                row["stimulus_blind_id"] for row in experimental
            ).values()))

            metadata = {row["stimulus_blind_id"]: row for row in key}
            for condition in design["conditions"]:
                selected = [row for row in key if row["condition"] == condition]
                self.assertEqual(len(selected), 72)
                self.assertEqual(set(Counter(row["prompt_category"] for row in selected).values()), {12})
                self.assertEqual(set(Counter(row["template_id"] for row in selected).values()), {4})
                self.assertEqual(set(Counter(row["payload_class"] for row in selected).values()), {18})
            by_slot = {}
            for row in experimental:
                by_slot.setdefault(row["panel_slot_id"], []).append(row)
            self.assertEqual(len(by_slot), 72)
            for rows in by_slot.values():
                self.assertEqual(len(rows), 24)
                conditions = Counter(metadata[row["stimulus_blind_id"]]["condition"] for row in rows)
                self.assertEqual(set(conditions.values()), {3})
                self.assertEqual(len({row["stimulus_blind_id"] for row in rows}), 24)

            subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "generate_synthetic_ratings.py"),
                 "--schedule", str(randomization / "participant_schedule.csv"),
                 "--blind-key", str(randomization / "blind_key.csv"),
                 "--output", str(ratings)],
                cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, text=True,
            )
            rating_rows = read_csv(ratings)
            self.assertEqual(len(rating_rows), 1728 * 7 + 144)
            outcome_rows = [row for row in rating_rows if row["item_type"] == "experimental_message"]
            self.assertEqual({int(row["rating"]) for row in outcome_rows} - set(range(1, 8)), set())
            self.assertTrue(all(row["synthetic_fixture"] == "true" for row in rating_rows))

            if shutil.which("Rscript"):
                output = directory / "r_validation"
                completed = subprocess.run(
                    ["Rscript", str(ROOT / "analysis" / "ordinal_mixed_model.R"),
                     "--data", str(ratings), "--output-dir", str(output), "--validate-only"],
                    cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue((output / "validation_summary.csv").exists())

    def test_power_simulation_is_reproducible_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            base = [
                sys.executable, str(ROOT / "power" / "simulate_power.py"),
                "--simulations", "30", "--seed", "42",
            ]
            subprocess.run(base + ["--output", str(first)], cwd=REPO_ROOT, check=True,
                           stdout=subprocess.PIPE, text=True)
            subprocess.run(base + ["--output", str(second)], cwd=REPO_ROOT, check=True,
                           stdout=subprocess.PIPE, text=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            rows = read_csv(first)
            self.assertEqual(len(rows), 6)
            for row in rows:
                probability = float(row["rejection_probability"])
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)
                self.assertEqual(int(row["simulations"]), 30)


if __name__ == "__main__":
    unittest.main()
