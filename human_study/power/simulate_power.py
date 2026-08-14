#!/usr/bin/env python3
"""Simulation-based planning power for clustered seven-category ratings.

The planning test compares stimulus-level mean ratings with a conservative Welch
normal approximation. Final inference uses the preregistered cumulative-link mixed
model; this dependency-free simulation is a transparent sensitivity tool, not a
replacement for that model.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import NormalDist, mean, variance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "power_scenarios.json"


def logistic_draw(rng):
    probability = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return math.log(probability / (1.0 - probability))


def ordinalize(latent, thresholds):
    for category, threshold in enumerate(thresholds, start=1):
        if latent <= threshold:
            return category
    return len(thresholds) + 1


def stimulus_means(effect, n_stimuli, n_participants, ratings_per_stimulus,
                   participant_effects, participant_shift, participant_sd,
                   stimulus_sd, thresholds, rng):
    del participant_sd  # effects are shared and already drawn at the requested SD
    output = []
    offsets = (0, 17, 37, 53, 67)
    if ratings_per_stimulus > len(offsets):
        raise ValueError("ratings_per_stimulus exceeds the frozen offset table")
    for stimulus_index in range(n_stimuli):
        stimulus_effect = rng.gauss(0.0, stimulus_sd)
        ratings = []
        for replicate in range(ratings_per_stimulus):
            participant = (
                stimulus_index + offsets[replicate] + participant_shift
            ) % n_participants
            latent = (
                effect + participant_effects[participant] + stimulus_effect
                + logistic_draw(rng)
            )
            ratings.append(ordinalize(latent, thresholds))
        output.append(mean(ratings))
    return output


def welch_p_value(first, second):
    difference = mean(second) - mean(first)
    standard_error = math.sqrt(variance(first) / len(first) + variance(second) / len(second))
    if standard_error == 0:
        return (0.0 if difference else 1.0), difference
    statistic = abs(difference / standard_error)
    p_value = 2.0 * (1.0 - NormalDist().cdf(statistic))
    return p_value, difference


def simulate_scenario(scenario, config, simulations, scenario_seed):
    rng = random.Random(scenario_seed)
    significant = 0
    differences = []
    odds_ratio = float(scenario["common_odds_ratio"])
    effect = math.log(odds_ratio)
    ratings_per_stimulus = int(
        scenario.get("ratings_per_stimulus", config["ratings_per_stimulus"])
    )
    participant_slots = int(
        scenario.get("participant_slots", config["participant_slots"])
    )
    if ratings_per_stimulus < 1 or ratings_per_stimulus > 5:
        raise ValueError("ratings_per_stimulus must be between 1 and 5")
    if participant_slots < ratings_per_stimulus:
        raise ValueError(
            "participant_slots must be at least ratings_per_stimulus"
        )
    for _ in range(simulations):
        participant_effects = [
            rng.gauss(0.0, scenario["participant_sd"])
            for _ in range(participant_slots)
        ]
        reference = stimulus_means(
            0.0, scenario["stimuli_per_condition"], participant_slots,
            ratings_per_stimulus, participant_effects, 0,
            scenario["participant_sd"], scenario["stimulus_sd"],
            config["thresholds"], rng,
        )
        target = stimulus_means(
            effect, scenario["stimuli_per_condition"], participant_slots,
            ratings_per_stimulus, participant_effects, 11,
            scenario["participant_sd"], scenario["stimulus_sd"],
            config["thresholds"], rng,
        )
        p_value, difference = welch_p_value(reference, target)
        significant += p_value < config["two_sided_alpha"]
        differences.append(difference)
    estimate = significant / simulations
    monte_carlo_se = math.sqrt(estimate * (1.0 - estimate) / simulations)
    return {
        "scenario_id": scenario["scenario_id"],
        "simulations": simulations,
        "seed": scenario_seed,
        "stimuli_per_condition": scenario["stimuli_per_condition"],
        "ratings_per_stimulus": ratings_per_stimulus,
        "participant_slots": participant_slots,
        "two_sided_alpha": config["two_sided_alpha"],
        "common_odds_ratio": odds_ratio,
        "participant_sd": scenario["participant_sd"],
        "stimulus_sd": scenario["stimulus_sd"],
        "rejection_probability": round(estimate, 6),
        "monte_carlo_se": round(monte_carlo_se, 6),
        "mean_observed_rating_difference": round(mean(differences), 6),
        "planning_method": "stimulus_cluster_welch_normal_approximation",
    }


def run(config, simulations=None, seed=None):
    simulations = int(simulations or config["default_simulations"])
    seed = int(config["seed"] if seed is None else seed)
    if simulations < 10:
        raise ValueError("at least 10 simulations are required")
    return [
        simulate_scenario(scenario, config, simulations, seed + index * 100003)
        for index, scenario in enumerate(config["scenarios"])
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = run(config, args.simulations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("{scenario_id}: {rejection_probability:.3f} (MC SE {monte_carlo_se:.3f})".format(**row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
