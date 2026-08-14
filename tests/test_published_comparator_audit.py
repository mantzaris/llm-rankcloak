import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_published_comparator import (
    ComparatorAuditError,
    audit_repository,
    classify_license,
    evaluate_fair_comparison,
    parse_pinned_requirements,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture()
def comparator_repository(tmp_path):
    repository = tmp_path / "published-comparator"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "remote", "add", "origin", "https://example.invalid/lm.git")

    (repository / "README.md").write_text("Historical comparator fixture\n")
    (repository / "encoder.py").write_text("# historical encoder fixture\n")
    (repository / "huffman.py").write_text(
        """
import heapq
import numpy as np

def build_min_heap(freqs, inds=None):
    inds = inds or range(len(freqs))
    values = [(freqs[ind], i, ind) for i, ind in enumerate(inds)]
    heapq.heapify(values)
    return values

def huffman_tree(heap):
    counter = len(heap)
    while len(heap) > 1:
        first_frequency, _, first = heapq.heappop(heap)
        second_frequency, _, second = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (first_frequency + second_frequency, counter, (first, second)),
        )
        counter += 1
    return heap[0][2]

def tv_huffman(code_tree, p):
    total = 0.0
    absent = np.ones_like(p)
    gap = 0.0
    stack = [(code_tree, 0)]
    while stack:
        node, depth = stack.pop()
        if type(node) is tuple:
            stack.append((node[0], depth + 1))
            stack.append((node[1], depth + 1))
        else:
            total += abs(p[node] - 2 ** (-depth))
            absent[node] = 0
            gap += p[node] * depth + p[node] * np.log2(p[node])
    return 0.5 * (total + np.sum(absent * p)), gap

def invert_code_tree(code_tree):
    return {}
""".lstrip()
    )
    (repository / "core.py").write_text(
        """
import numpy as np
from huffman import build_min_heap, huffman_tree, tv_huffman, invert_code_tree

class Sender:
    def __init__(self, max_sequence_length=80):
        self.max_sequence_length = max_sequence_length

    def embed_bits(self, coin_flips):
        heap = build_min_heap([0.5, 0.5])
        hc = huffman_tree(heap)
        if tv_huffman(hc, [0.5, 0.5])[0] < self.tv_threshold:
            return coin_flips
        return self.random.choice(self.lm.vocabulary_size, p=p)

class Receiver:
    def recover_bits(self, token_inds, remaining_bits):
        hc = huffman_tree([])
        return token_inds

if __name__ == '__main__':
    from gptlm import GptLanguageModel
    cipher_text_length = 32
    tv_threshold = 0.08
    alice = Sender(seed=123)
""".lstrip()
    )
    (repository / "gptlm.py").write_text(
        """
import tensorflow as tf
spec_path = "./external/gpt-2/src/model.py"

class GptLanguageModel:
    def __init__(self, model_name='117M'):
        self.model_name = model_name
""".lstrip()
    )
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Audit Fixture",
        "-c",
        "user.email=audit@example.invalid",
        "commit",
        "-q",
        "-m",
        "historical source",
    )
    historical_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "tag", "acl-2019")

    (repository / "LICENSE").write_text(
        """
The MIT License
Copyright (c) 2019 Fixture Author
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to deal in the Software without restriction.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
""".lstrip()
    )
    (repository / "requirements.txt").write_text(
        "numpy==1.16.1\ntensorflow==1.12.0\ntorch==1.0.1.post2\n"
    )
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Audit Fixture",
        "-c",
        "user.email=audit@example.invalid",
        "commit",
        "-q",
        "-m",
        "later license and requirements",
    )
    current_commit = _git(repository, "rev-parse", "HEAD")
    _git(
        repository,
        "update-ref",
        "refs/remotes/origin/master",
        current_commit,
    )
    _git(repository, "checkout", "-q", "--detach", historical_commit)
    return repository, historical_commit, current_commit


def test_license_and_requirement_classification():
    license_result = classify_license(
        "Copyright (c) 2019 Author\n"
        "Permission is hereby granted, free of charge, to use without restriction.\n"
        'THE SOFTWARE IS PROVIDED "AS IS".\n'
    )
    assert license_result["classification"] == "MIT"
    assert license_result["legal_advice"] is False
    requirements = parse_pinned_requirements(
        "# comment\nTensorFlow==1.12.0\nnumpy==1.16.1\nunpinned\n"
    )
    assert requirements == {"tensorflow": "1.12.0", "numpy": "1.16.1"}


def test_fair_comparison_requires_every_prespecified_condition():
    failed = evaluate_fair_comparison({})
    assert (
        failed[
            "numeric_patient_huffman_comparison_defensible_without_rewriting_official_method"
        ]
        is False
    )
    assert len(failed["unmet_required_criteria"]) == 7
    passed = evaluate_fair_comparison(
        {
            "same_prompts": True,
            "same_model": True,
            "same_payloads": True,
            "matched_controls": True,
            "recovery_measured": True,
            "same_evaluation": True,
            "official_end_to_end_runnable": True,
        }
    )
    assert passed["decision"] == "defensible"


def test_fixture_audit_reproduces_kernel_and_dependency_failure(
    comparator_repository,
):
    repository, historical_commit, current_commit = comparator_repository
    report = audit_repository(
        repository,
        expected_historical_commit=historical_commit,
        historical_tag="acl-2019",
        current_ref="origin/master",
        expected_current_commit=current_commit,
        python_executable=sys.executable,
    )
    assert report["integrity"]["all_provenance_checks_passed"]
    assert report["integrity"]["external_checkout_unchanged"]
    assert report["historical_tag_tree"]["license_file_present"] is False
    assert report["historical_tag_tree"]["requirements_file_present"] is False
    assert report["later_current_ref"]["license"]["classification"] == "MIT"
    assert (
        report["later_current_ref"]["selected_pinned_requirements"][
            "tensorflow"
        ]
        == "1.12.0"
    )
    assert report["reproduction"]["standalone_huffman_dyadic_check"]["passed"]
    failure = report["reproduction"]["official_entry_point"]["failure"]
    assert failure["type"] == "ModuleNotFoundError"
    assert "tensorflow" in failure["message"]
    assert (
        report["overall_conclusion"]["numeric_patient_huffman_baseline_approved"]
        is False
    )
    assert _git(repository, "status", "--porcelain") == ""


def test_provenance_mismatch_is_machine_visible(comparator_repository):
    repository, _, current_commit = comparator_repository
    report = audit_repository(
        repository,
        expected_historical_commit="0" * 40,
        historical_tag="acl-2019",
        current_ref="origin/master",
        expected_current_commit=current_commit,
        python_executable=sys.executable,
        run_probes=False,
    )
    assert not report["integrity"]["all_provenance_checks_passed"]
    assert not report["provenance"]["checks"][
        "head_matches_expected_historical_commit"
    ]
    assert report["reproduction"]["standalone_huffman_dyadic_check"]["skipped"]


def test_report_writer_refuses_unapproved_overwrite(tmp_path):
    output = tmp_path / "audit.json"
    write_report({"status": "first"}, output)
    assert json.loads(output.read_text()) == {"status": "first"}
    with pytest.raises(ComparatorAuditError, match="overwrite"):
        write_report({"status": "second"}, output)
    write_report({"status": "second"}, output, overwrite=True)
    assert json.loads(output.read_text()) == {"status": "second"}


def test_cli_emits_and_writes_machine_json(
    comparator_repository, tmp_path
):
    repository, historical_commit, current_commit = comparator_repository
    output = tmp_path / "comparator-audit.json"
    process = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_published_comparator.py"),
            "--repository",
            str(repository),
            "--expected-historical-commit",
            historical_commit,
            "--expected-current-commit",
            current_commit,
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    stdout_report = json.loads(process.stdout)
    file_report = json.loads(output.read_text())
    assert stdout_report == file_report
    assert file_report["schema_version"].endswith("-v1")
    assert (
        file_report["audit_policy"]["official_method_modified_or_vendored"]
        is False
    )
