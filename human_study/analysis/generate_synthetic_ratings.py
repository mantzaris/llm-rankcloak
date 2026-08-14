#!/usr/bin/env python3
"""Generate synthetic long-format ratings from a blinded fixture schedule."""

import argparse
import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUMENT = ROOT / "survey" / "instrument.json"

BASE_EFFECTS = {
    "human_written_control": 0.65,
    "ordinary_llm_control": 0.45,
    "direct_subword_calgacus": -0.55,
    "rankcloak_ascii_b8": 0.10,
    "rankcloak_ascii_b16": -0.10,
    "rankcloak_hex_nibble": -0.05,
    "rankcloak_segmented_forced_span": -0.60,
    "rankcloak_segmented_full_message": 0.15,
    "rankcloak_segmented_single_topic": -0.60,
    "rankcloak_segmented_multi_topic": 0.15,
}

OUTPUT_FIELDS = [
    "response_id", "participant_slot_id", "presentation_order", "item_type",
    "stimulus_blind_id", "scale_id", "rating", "attention_check_id",
    "expected_response", "condition", "prompt_category", "template_id",
    "payload_id", "payload_class", "model_family", "presentation_scope",
    "response_time_ms", "synthetic_fixture", "include_primary",
]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clamp_rating(value):
    return max(1, min(7, int(round(value))))


def generate(schedule, blind_key, instrument, seed):
    rng = random.Random(seed)
    key = {row["stimulus_blind_id"]: row for row in blind_key}
    participant_effects = {}
    stimulus_effects = {}
    rows = []
    response_index = 0
    for item in schedule:
        participant = item["panel_slot_id"]
        participant_effects.setdefault(participant, rng.gauss(0.0, 0.45))
        if item["item_type"] == "attention_check":
            response_index += 1
            rows.append({
                "response_id": "SYN{:06d}".format(response_index),
                "participant_slot_id": participant,
                "presentation_order": item["presentation_order"],
                "item_type": "attention_check",
                "stimulus_blind_id": "",
                "scale_id": "attention_check",
                "rating": item["expected_response"],
                "attention_check_id": item["attention_check_id"],
                "expected_response": item["expected_response"],
                "condition": "",
                "prompt_category": "",
                "template_id": "",
                "payload_id": "",
                "payload_class": "",
                "model_family": "",
                "presentation_scope": "",
                "response_time_ms": str(rng.randint(1800, 5000)),
                "synthetic_fixture": "true",
                "include_primary": "false",
            })
            continue
        blind_id = item["stimulus_blind_id"]
        metadata = key[blind_id]
        stimulus_effects.setdefault(blind_id, rng.gauss(0.0, 0.4))
        condition_effect = BASE_EFFECTS.get(metadata["condition"], 0.0)
        for scale in instrument["scales"]:
            response_index += 1
            direction = -1.0 if scale["scale_id"] == "suspiciousness" else 1.0
            latent = (
                4.0 + direction * condition_effect
                + participant_effects[participant]
                + stimulus_effects[blind_id]
                + rng.gauss(0.0, 0.9)
            )
            rows.append({
                "response_id": "SYN{:06d}".format(response_index),
                "participant_slot_id": participant,
                "presentation_order": item["presentation_order"],
                "item_type": "experimental_message",
                "stimulus_blind_id": blind_id,
                "scale_id": scale["scale_id"],
                "rating": str(clamp_rating(latent)),
                "attention_check_id": "",
                "expected_response": "",
                "condition": metadata["condition"],
                "prompt_category": metadata["prompt_category"],
                "template_id": metadata["template_id"],
                "payload_id": metadata["payload_id"],
                "payload_class": metadata["payload_class"],
                "model_family": metadata["model_family"],
                "presentation_scope": metadata["presentation_scope"],
                "response_time_ms": str(rng.randint(5000, 30000)),
                "synthetic_fixture": "true",
                "include_primary": "true",
            })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    schedule = read_csv(args.schedule)
    key = read_csv(args.blind_key)
    instrument = json.loads(args.instrument.read_text(encoding="utf-8"))
    rows = generate(schedule, key, instrument, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} synthetic response rows to {}".format(len(rows), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
