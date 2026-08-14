#!/usr/bin/env python3
"""Generate a balanced synthetic-only stimulus pool for pipeline testing."""

import argparse
import csv
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = PACKAGE_ROOT / "config" / "design.json"
DEFAULT_SPEC = PACKAGE_ROOT / "fixtures" / "synthetic_fixture_spec.json"

FIELDS = [
    "stimulus_id", "condition", "prompt_category", "template_id", "payload_id",
    "payload_class", "model_family", "presentation_scope", "message_text",
    "synthetic_fixture", "license_status", "safety_screen_status",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_rows(design, spec):
    rows = []
    models = design["model_families"]
    replicates = int(spec["candidate_replicates_per_stratum"])
    counter = 0
    for condition in design["conditions"]:
        for category_index, category in enumerate(design["prompt_categories"]):
            for template_number in range(1, design["templates_per_category"] + 1):
                template_id = "{}_template_{}".format(category, template_number)
                for payload_index, payload_class in enumerate(design["eligible_payload_classes"]):
                    if condition == "human_written_control":
                        model_family = "human"
                    else:
                        model_family = models[
                            (category_index + template_number - 1 + payload_index) % len(models)
                        ]
                    presentation_scope = (
                        "forced_span"
                        if condition == "rankcloak_segmented_forced_span"
                        else "full_message"
                    )
                    for replicate in range(1, replicates + 1):
                        counter += 1
                        stimulus_id = "FIX{:05d}".format(counter)
                        payload_id = "{}_{}_{}_r{}".format(
                            payload_class, category_index + 1, template_number, replicate
                        )
                        text = (
                            "{}: candidate {} for {} / {} / {} / {}. This neutral "
                            "placeholder validates balancing and must not be used as a "
                            "research stimulus."
                        ).format(
                            spec["text_prefix"], replicate, condition, category,
                            template_id, payload_class,
                        )
                        rows.append({
                            "stimulus_id": stimulus_id,
                            "condition": condition,
                            "prompt_category": category,
                            "template_id": template_id,
                            "payload_id": payload_id,
                            "payload_class": payload_class,
                            "model_family": model_family,
                            "presentation_scope": presentation_scope,
                            "message_text": text,
                            "synthetic_fixture": "true",
                            "license_status": "synthetic_fixture",
                            "safety_screen_status": "fixture_only",
                        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    design = load_json(args.design)
    spec = load_json(args.spec)
    rows = generate_rows(design, spec)
    expected = (
        len(design["conditions"]) * len(design["prompt_categories"])
        * design["templates_per_category"] * len(design["eligible_payload_classes"])
        * design["candidate_replicates_per_stratum"]
    )
    if len(rows) != expected:
        raise RuntimeError("candidate count mismatch: {} != {}".format(len(rows), expected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} synthetic candidates to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
