#!/usr/bin/env python3
"""Select balanced stimuli and build blinded three-rating schedules."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "config" / "design.json"
DEFAULT_CHECKS = ROOT / "survey" / "attention_checks.json"
REQUIRED = {
    "stimulus_id", "condition", "prompt_category", "template_id", "payload_id",
    "payload_class", "model_family", "presentation_scope", "message_text",
    "license_status", "safety_screen_status",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError("candidate manifest missing {}".format(sorted(missing)))
        rows = list(reader)
    identifiers = [row["stimulus_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate stimulus IDs are not unique")
    return rows


def select_rows(candidates, design, rng):
    groups = defaultdict(list)
    allowed = set(design["conditions"])
    for row in candidates:
        if row["condition"] not in allowed:
            raise ValueError("unexpected condition {}".format(row["condition"]))
        key = (
            row["condition"], row["prompt_category"], row["template_id"],
            row["payload_class"],
        )
        groups[key].append(row)
    selected = []
    for condition in design["conditions"]:
        for category in design["prompt_categories"]:
            for number in range(1, design["templates_per_category"] + 1):
                template = "{}_template_{}".format(category, number)
                for payload_class in design["eligible_payload_classes"]:
                    key = (condition, category, template, payload_class)
                    options = list(groups.get(key, []))
                    if not options:
                        raise ValueError("empty selection stratum {}".format(key))
                    rng.shuffle(options)
                    selected.append(options[0])
    expected = len(design["conditions"]) * design["stimuli_per_condition"]
    if len(selected) != expected:
        raise RuntimeError("selected {} rows, expected {}".format(len(selected), expected))
    return selected


def blind_mapping(selected, rng):
    shuffled = list(selected)
    rng.shuffle(shuffled)
    return {
        row["stimulus_id"]: "B{:04d}".format(index)
        for index, row in enumerate(shuffled, start=1)
    }


def nonadjacent_shuffle(items, rng):
    for _ in range(500):
        remaining = list(items)
        output = []
        last = None
        while remaining:
            valid = [i for i, item in enumerate(remaining) if item["_condition"] != last]
            if not valid:
                break
            item = remaining.pop(rng.choice(valid))
            output.append(item)
            last = item["_condition"]
        if len(output) == len(items):
            return output
    raise RuntimeError("could not satisfy nonadjacent condition order")


def experimental_schedules(selected, blind, design, rng):
    by_condition = defaultdict(list)
    for row in selected:
        item = dict(row)
        item["stimulus_blind_id"] = blind[row["stimulus_id"]]
        by_condition[row["condition"]].append(item)
    shifts = (0, 17, 37)
    if len(shifts) != design["items_per_condition_per_panel_slot"]:
        raise ValueError("allocator is frozen for three items per condition")
    schedules = {i: [] for i in range(design["panel_slots"])}
    for condition in design["conditions"]:
        rows = list(by_condition[condition])
        rng.shuffle(rows)
        if len(rows) != design["panel_slots"]:
            raise ValueError("{} has {} selected rows".format(condition, len(rows)))
        for panel_index in range(design["panel_slots"]):
            for replicate, shift in enumerate(shifts, start=1):
                row = rows[(panel_index + shift) % len(rows)]
                schedules[panel_index].append({
                    "_condition": condition,
                    "panel_slot_id": "P{:03d}".format(panel_index + 1),
                    "item_type": "experimental_message",
                    "stimulus_blind_id": row["stimulus_blind_id"],
                    "topic_label": row["prompt_category"].replace("_", " "),
                    "message_text": row["message_text"],
                    "attention_check_id": "",
                    "expected_response": "",
                    "allocation_replicate": str(replicate),
                })
    return schedules


def complete_schedules(schedules, checks, design, rng):
    if len(checks["checks"]) != design["attention_checks_per_panel_slot"]:
        raise ValueError("attention-check count differs from design")
    output = []
    for panel_index in range(design["panel_slots"]):
        rows = nonadjacent_shuffle(schedules[panel_index], rng)
        positions = sorted(rng.sample(range(4, len(rows) + 2), len(checks["checks"])))
        for position, check in zip(positions, checks["checks"]):
            rows.insert(position, {
                "_condition": "",
                "panel_slot_id": "P{:03d}".format(panel_index + 1),
                "item_type": "attention_check",
                "stimulus_blind_id": "",
                "topic_label": "instruction check",
                "message_text": check["prompt"],
                "attention_check_id": check["attention_check_id"],
                "expected_response": str(check["expected_response"]),
                "allocation_replicate": "",
            })
        for order, row in enumerate(rows, start=1):
            public = dict(row)
            public.pop("_condition")
            public["presentation_order"] = str(order)
            output.append(public)
    return output


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balance(selected, design):
    result = {}
    for condition in design["conditions"]:
        rows = [row for row in selected if row["condition"] == condition]
        result[condition] = {
            "total": len(rows),
            "prompt_category": dict(Counter(row["prompt_category"] for row in rows)),
            "template_id": dict(Counter(row["template_id"] for row in rows)),
            "payload_class": dict(Counter(row["payload_class"] for row in rows)),
            "model_family": dict(Counter(row["model_family"] for row in rows)),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--attention-checks", type=Path, default=DEFAULT_CHECKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    design = read_json(args.design)
    checks = read_json(args.attention_checks)
    candidates = read_rows(args.candidates)
    rng = random.Random(design["random_seed"])
    selected = select_rows(candidates, design, rng)
    blind = blind_mapping(selected, rng)
    schedule = complete_schedules(
        experimental_schedules(selected, blind, design, rng), checks, design, rng
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    public = [{
        "stimulus_blind_id": blind[row["stimulus_id"]],
        "topic_label": row["prompt_category"].replace("_", " "),
        "message_text": row["message_text"],
    } for row in selected]
    key = [{
        "stimulus_blind_id": blind[row["stimulus_id"]],
        **{field: row[field] for field in sorted(REQUIRED)},
    } for row in selected]
    write_csv(
        args.output_dir / "selected_stimuli_blinded.csv", public,
        ["stimulus_blind_id", "topic_label", "message_text"],
    )
    write_csv(
        args.output_dir / "blind_key.csv", key,
        ["stimulus_blind_id"] + sorted(REQUIRED),
    )
    schedule_fields = [
        "panel_slot_id", "presentation_order", "item_type", "stimulus_blind_id",
        "topic_label", "message_text", "attention_check_id", "expected_response",
        "allocation_replicate",
    ]
    write_csv(args.output_dir / "participant_schedule.csv", schedule, schedule_fields)

    audit = {
        "schema_version": "1.0.0",
        "status": "SYNTHETIC_DRY_RUN" if all(
            row.get("synthetic_fixture") == "true" for row in candidates
        ) else "DRAFT_REAL_STIMULUS_SELECTION_NOT_AUTHORIZED",
        "seed": design["random_seed"],
        "candidate_input_sha256": file_hash(args.candidates),
        "design_sha256": file_hash(args.design),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "experimental_exposures": sum(
            row["item_type"] == "experimental_message" for row in schedule
        ),
        "attention_check_rows": sum(row["item_type"] == "attention_check" for row in schedule),
        "balance": balance(selected, design),
    }
    for name in ("selected_stimuli_blinded.csv", "blind_key.csv", "participant_schedule.csv"):
        audit[name + "_sha256"] = file_hash(args.output_dir / name)
    (args.output_dir / "design_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("selected {}; scheduled {} exposures and {} checks".format(
        len(selected), audit["experimental_exposures"], audit["attention_check_rows"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
