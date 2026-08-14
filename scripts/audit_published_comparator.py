#!/usr/bin/env python3
"""Read-only provenance and reproducibility audit for patient-Huffman code.

The audit intentionally does not initialize submodules, install TensorFlow,
download GPT-2 assets, or alter the external checkout.  Its JSON output
distinguishes a successful standalone mathematical-kernel check from an
end-to-end cover-generation comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = PROJECT_ROOT / "external_sources" / "lm-steganography"
DEFAULT_HISTORICAL_TAG = "acl-2019"
DEFAULT_HISTORICAL_COMMIT = "0f0c41061242688e1830e16de5718f04662b10b2"
DEFAULT_CURRENT_REF = "origin/master"
DEFAULT_CURRENT_COMMIT = "f0ee4b0097a90ffe368c52adf61f2601970c0435"
SCHEMA_VERSION = "rankcloak-published-comparator-audit-v1"


class ComparatorAuditError(RuntimeError):
    """Raised when audit provenance or output handling is invalid."""


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    cwd: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def _run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float = 20.0,
    environment: Mapping[str, str] | None = None,
) -> CommandResult:
    env = os.environ.copy()
    if environment:
        env.update({str(key): str(value) for key, value in environment.items()})
    try:
        completed = subprocess.run(
            list(map(str, argv)),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        return CommandResult(
            argv=list(map(str, argv)),
            cwd=str(cwd.resolve()),
            returncode=int(completed.returncode),
            stdout=completed.stdout[-8_000:],
            stderr=completed.stderr[-8_000:],
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return CommandResult(
            argv=list(map(str, argv)),
            cwd=str(cwd.resolve()),
            returncode=None,
            stdout=stdout[-8_000:],
            stderr=stderr[-8_000:],
            timed_out=True,
        )


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    timeout_seconds: float = 20.0,
) -> str:
    result = _run_command(
        ["git", *arguments], cwd=repository, timeout_seconds=timeout_seconds
    )
    if check and result.returncode != 0:
        raise ComparatorAuditError(
            "git {} failed in {}: {}".format(
                " ".join(arguments), repository, result.stderr.strip()
            )
        )
    return result.stdout.strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_file(repository: Path, revision: str, path: str) -> str:
    return _git(repository, "show", f"{revision}:{path}")


def _git_file_metadata(
    repository: Path, revision: str, path: str
) -> dict[str, Any]:
    value = _git_file(repository, revision, path)
    return {
        "path": path,
        "git_blob": _git(repository, "rev-parse", f"{revision}:{path}"),
        "sha256": _sha256_text(value),
        "bytes_utf8": len(value.encode("utf-8")),
    }


def _commit_metadata(repository: Path, revision: str) -> dict[str, str]:
    fields = _git(
        repository,
        "show",
        "-s",
        "--format=%H%n%aI%n%cI%n%s",
        revision,
    ).splitlines()
    if len(fields) < 4:
        raise ComparatorAuditError(f"Could not parse commit metadata for {revision}")
    return {
        "commit": fields[0],
        "author_date": fields[1],
        "committer_date": fields[2],
        "subject": "\n".join(fields[3:]),
    }


def classify_license(text: str) -> dict[str, Any]:
    """Classify only the explicit license text; this is not legal advice."""

    normalized = " ".join(text.lower().split())
    mit_markers = (
        "permission is hereby granted, free of charge",
        "the software is provided \"as is\"",
        "without restriction",
    )
    is_mit = all(marker in normalized for marker in mit_markers)
    copyright_match = re.search(
        r"copyright\s*(?:\(c\))?\s*([^\n]+)", text, flags=re.IGNORECASE
    )
    return {
        "classification": "MIT" if is_mit else "unclassified",
        "mit_markers_present": is_mit,
        "copyright_line": (
            copyright_match.group(0).strip() if copyright_match else None
        ),
        "sha256": _sha256_text(text),
        "legal_advice": False,
    }


def parse_pinned_requirements(text: str) -> dict[str, str]:
    """Parse simple exact pins while retaining only package/version facts."""

    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        result[name.strip().lower()] = version.strip()
    return result


def _line_number(source: str, needle: str) -> int | None:
    for index, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return index
    return None


def _source_method_evidence(
    repository: Path, historical_revision: str
) -> dict[str, Any]:
    core = _git_file(repository, historical_revision, "core.py")
    huffman = _git_file(repository, historical_revision, "huffman.py")
    gptlm = _git_file(repository, historical_revision, "gptlm.py")
    return {
        "method_name": "patient-Huffman",
        "historical_source_locations": {
            "sender_embed_bits": {
                "file": "core.py",
                "line": _line_number(core, "def embed_bits"),
            },
            "per_step_huffman_tree": {
                "file": "core.py",
                "line": _line_number(core, "hc = huffman_tree(heap)"),
            },
            "tv_acceptance_threshold": {
                "file": "core.py",
                "line": _line_number(
                    core, "tv_huffman(hc, p)[0] < self.tv_threshold"
                ),
            },
            "ordinary_sampling_when_not_accepted": {
                "file": "core.py",
                "line": _line_number(
                    core, "self.random.choice(self.lm.vocabulary_size, p=p)"
                ),
            },
            "receiver_rebuild_and_recovery": {
                "file": "core.py",
                "line": _line_number(core, "def recover_bits"),
            },
            "total_variation_implementation": {
                "file": "huffman.py",
                "line": _line_number(huffman, "def tv_huffman"),
            },
            "tensorflow_import": {
                "file": "gptlm.py",
                "line": _line_number(gptlm, "import tensorflow as tf"),
            },
            "gpt2_model_loader": {
                "file": "gptlm.py",
                "line": _line_number(
                    gptlm,
                    "./external/gpt-2/src/model.py",
                ),
            },
            "gpt2_117m_default": {
                "file": "gptlm.py",
                "line": _line_number(gptlm, "model_name='117M'"),
            },
        },
        "historical_example_configuration": {
            "ciphertext_bits": 32
            if "cipher_text_length = 32" in core
            else None,
            "tv_threshold": 0.08 if "tv_threshold = 0.08" in core else None,
            "sender_seed": 123 if "seed=123" in core else None,
            "maximum_sequence_length_default": 80
            if "max_sequence_length=80" in core
            else None,
            "conditioning": (
                "GPT-2 end-of-text/SOS prefix; the historical core example "
                "does not accept RankCloak-style topic prompts"
            ),
        },
        "method_summary": (
            "At each autoregressive step the sender constructs a Huffman tree "
            "from the model conditional distribution. It consumes payload bits "
            "only when the induced Huffman distribution is below a total-variation "
            "threshold; otherwise it samples normally. The receiver reconstructs "
            "the same tree from the same prefix and maps observed tokens back to bits."
        ),
    }


def _exception_summary(stderr: str) -> dict[str, str | None]:
    matches = re.findall(
        r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.*)$",
        stderr,
        flags=re.MULTILINE,
    )
    if not matches:
        return {"type": None, "message": None}
    exception_type, message = matches[-1]
    return {"type": exception_type, "message": message.strip()}


def run_huffman_probe(
    repository: Path,
    *,
    python_executable: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Exercise the official standalone Huffman functions on a dyadic p."""

    source = (
        "import json, numpy as np; "
        "from huffman import build_min_heap, huffman_tree, tv_huffman; "
        "p=np.asarray([0.5,0.25,0.125,0.125], dtype=float); "
        "tree=huffman_tree(build_min_heap(p)); "
        "tv,gap=tv_huffman(tree,p); "
        "print(json.dumps({'distribution':p.tolist(),"
        "'code_tree':tree,'total_variation':float(tv),"
        "'cross_entropy_gap':float(gap)},sort_keys=True))"
    )
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = _run_command(
        [python_executable, "-B", "-c", source],
        cwd=repository,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    parsed: dict[str, Any] | None = None
    if result.returncode == 0:
        nonempty = [line for line in result.stdout.splitlines() if line.strip()]
        try:
            parsed = json.loads(nonempty[-1]) if nonempty else None
        except json.JSONDecodeError:
            parsed = None
    total_variation = parsed.get("total_variation") if parsed else None
    return {
        "purpose": (
            "mathematical-kernel smoke check only; not cover generation, "
            "recovery, naturalness, or detectability evidence"
        ),
        "distribution": [0.5, 0.25, 0.125, 0.125],
        "expected_total_variation": 0.0,
        "observed": parsed,
        "passed": (
            result.returncode == 0
            and total_variation is not None
            and abs(float(total_variation)) <= 1e-12
        ),
        "command": asdict(result),
    }


def run_official_entry_probe(
    repository: Path,
    *,
    python_executable: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Attempt the official historical core entry point without mutation."""

    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = _run_command(
        [python_executable, "-B", "core.py"],
        cwd=repository,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    exception = _exception_summary(result.stderr)
    return {
        "purpose": "official historical end-to-end illustrative entry point",
        "completed": result.returncode == 0,
        "failure": (
            None
            if result.returncode == 0
            else {
                **exception,
                "stage": (
                    "dependency_import"
                    if exception["type"] == "ModuleNotFoundError"
                    else "entry_point"
                ),
            }
        ),
        "command": asdict(result),
        "network_or_install_permitted": False,
    }


def evaluate_fair_comparison(context: Mapping[str, Any]) -> dict[str, Any]:
    """Apply prespecified comparability criteria to the available implementation."""

    criteria = [
        {
            "criterion": "same_prompt_instances_and_conditioning",
            "required": True,
            "met": bool(context.get("same_prompts")),
            "rationale": (
                "Prompt/topic conditioning must be identical or explicitly "
                "counterbalanced so prose-quality differences are not prompt effects."
            ),
        },
        {
            "criterion": "same_or_comparable_model_scale_and_tokenizer",
            "required": True,
            "met": bool(context.get("same_model")),
            "rationale": (
                "The codec comparison should use the same underlying LM when "
                "possible; GPT-2 117M versus modern 7--8B instruction models is "
                "a model-era and scale confound."
            ),
        },
        {
            "criterion": "same_independent_payload_bits_and_assignments",
            "required": True,
            "met": bool(context.get("same_payloads")),
            "rationale": (
                "Methods must receive the same payload bits, with independent "
                "payload identifiers retained as the statistical units."
            ),
        },
        {
            "criterion": "matched_controls_lengths_and_stopping",
            "required": True,
            "met": bool(context.get("matched_controls")),
            "rationale": (
                "Ordinary-LM controls, cover lengths, truncation, and stopping "
                "rules must be matched or adjusted prospectively."
            ),
        },
        {
            "criterion": "end_to_end_recovery_under_declared_assumptions",
            "required": True,
            "met": bool(context.get("recovery_measured")),
            "rationale": (
                "A successful local Huffman calculation is insufficient; encoding, "
                "text serialization, decoding, and exact recovery must be measured."
            ),
        },
        {
            "criterion": "same_blinded_quality_and_steganalysis_evaluation",
            "required": True,
            "met": bool(context.get("same_evaluation")),
            "rationale": (
                "Human ratings, automated quality measures, and detectors must be "
                "applied blindly to length- and prompt-matched outputs."
            ),
        },
        {
            "criterion": "official_or_validated_faithful_implementation",
            "required": True,
            "met": bool(context.get("official_end_to_end_runnable")),
            "rationale": (
                "A port or rewrite must be separately validated against official "
                "outputs before being called a published-method comparator."
            ),
        },
    ]
    unmet = [item["criterion"] for item in criteria if item["required"] and not item["met"]]
    decisive_blockers: list[str] = []
    if not context.get("official_end_to_end_runnable"):
        decisive_blockers.append(
            "The pinned official historical implementation is not end-to-end "
            "runnable with the present dependency and model assets."
        )
    if not context.get("same_model") or not context.get("same_prompts"):
        decisive_blockers.append(
            "Running the official GPT-2 117M/SOS configuration as-is would confound "
            "the steganographic method with model scale, tokenizer, and prompting."
        )
    if not context.get("same_payloads") or not context.get("matched_controls"):
        decisive_blockers.append(
            "No prospectively matched payload/control/length corpus has been "
            "generated for the comparator."
        )
    defensible = not unmet
    return {
        "criteria": criteria,
        "required_criteria_met": len(criteria) - len(unmet),
        "required_criteria_total": len(criteria),
        "unmet_required_criteria": unmet,
        "numeric_patient_huffman_comparison_defensible_without_rewriting_official_method": defensible,
        "decision": (
            "defensible"
            if defensible
            else "not_defensible_without_a_validated_reimplementation_or_historical_runtime"
        ),
        "decisive_blockers": decisive_blockers,
        "permitted_claim": (
            "The official standalone Huffman kernel can be reported as an audit "
            "smoke check if it passes. It cannot be reported as a numeric cover-text "
            "baseline or as evidence of comparative recovery, quality, or detectability."
        ),
        "future_path": [
            (
                "Reconstruct the exact historical TensorFlow/GPT-2 117M environment "
                "and model assets in an isolated, licensed archive, then run its "
                "native SOS-conditioned study as a historically faithful but "
                "model-confounded comparison."
            ),
            (
                "Alternatively, implement patient-Huffman on the same pinned models "
                "and prompts as RankCloak, validate it against official small test "
                "vectors/outputs, and label it a faithful reimplementation rather "
                "than the untouched official code."
            ),
        ],
    }


def audit_repository(
    repository: Path,
    *,
    expected_historical_commit: str = DEFAULT_HISTORICAL_COMMIT,
    historical_tag: str = DEFAULT_HISTORICAL_TAG,
    current_ref: str = DEFAULT_CURRENT_REF,
    expected_current_commit: str = DEFAULT_CURRENT_COMMIT,
    python_executable: str = sys.executable,
    timeout_seconds: float = 20.0,
    run_probes: bool = True,
) -> dict[str, Any]:
    """Create a machine-readable, read-only comparator audit."""

    repository = Path(repository).resolve()
    if not (repository / ".git").exists():
        raise ComparatorAuditError(f"Not a Git checkout: {repository}")
    status_before = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    head = _git(repository, "rev-parse", "HEAD")
    tag_commit = _git(repository, "rev-parse", f"{historical_tag}^{{commit}}")
    tag_object = _git(repository, "rev-parse", historical_tag)
    exact_tag = _git(
        repository, "describe", "--tags", "--exact-match", check=False
    )
    current_commit = _git(repository, "rev-parse", current_ref)
    origin_url = _git(
        repository, "remote", "get-url", "origin", check=False
    )
    historical_files = set(
        _git(repository, "ls-tree", "-r", "--name-only", head).splitlines()
    )
    current_files = set(
        _git(
            repository, "ls-tree", "-r", "--name-only", current_ref
        ).splitlines()
    )

    later_license_text = (
        _git_file(repository, current_ref, "LICENSE")
        if "LICENSE" in current_files
        else ""
    )
    later_requirements_text = (
        _git_file(repository, current_ref, "requirements.txt")
        if "requirements.txt" in current_files
        else ""
    )
    requirements = parse_pinned_requirements(later_requirements_text)
    changed_names = _git(
        repository,
        "diff",
        "--name-only",
        f"{head}..{current_ref}",
        check=False,
    ).splitlines()
    gitlink_line = _git(
        repository,
        "ls-tree",
        head,
        "external/gpt-2",
        check=False,
    )
    gitlink_match = re.match(
        r"160000\s+commit\s+([0-9a-f]{40})\s+external/gpt-2",
        gitlink_line,
    )
    submodule_status = _git(
        repository, "submodule", "status", check=False
    )
    submodule_initialized = bool(submodule_status) and not submodule_status.startswith(
        "-"
    )
    gpt2_path = repository / "external" / "gpt-2"
    model_path = gpt2_path / "models" / "117M"
    model_asset_names = (
        sorted(
            str(path.relative_to(repository))
            for path in model_path.rglob("*")
            if path.is_file()
        )
        if model_path.is_dir()
        else []
    )
    checkpoint_assets = [
        name
        for name in model_asset_names
        if Path(name).name == "checkpoint"
        or Path(name).suffix in {".index", ".meta"}
        or ".data-" in Path(name).name
    ]
    method_evidence = _source_method_evidence(repository, head)
    historical_hashes = {
        path: _git_file_metadata(repository, head, path)
        for path in ("core.py", "huffman.py", "gptlm.py", "encoder.py")
        if path in historical_files
    }

    if run_probes:
        huffman_probe = run_huffman_probe(
            repository,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        entry_probe = run_official_entry_probe(
            repository,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
    else:
        huffman_probe = {
            "passed": None,
            "skipped": True,
            "reason": "requested by --skip-probes",
        }
        entry_probe = {
            "completed": None,
            "skipped": True,
            "reason": "requested by --skip-probes",
        }

    status_after = _git(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    head_after = _git(repository, "rev-parse", "HEAD")
    provenance_checks = {
        "head_matches_expected_historical_commit": head
        == expected_historical_commit,
        "tag_resolves_to_expected_historical_commit": tag_commit
        == expected_historical_commit,
        "checkout_exactly_at_historical_tag": exact_tag == historical_tag,
        "current_ref_matches_expected_commit": current_commit
        == expected_current_commit,
        "clean_before": status_before == "",
        "clean_after": status_after == "",
        "head_unchanged_by_audit": head_after == head,
        "status_unchanged_by_audit": status_after == status_before,
    }
    license_facts = (
        classify_license(later_license_text)
        if later_license_text
        else {
            "classification": "missing",
            "mit_markers_present": False,
            "copyright_line": None,
            "sha256": None,
            "legal_advice": False,
        }
    )
    historical_license_missing = not any(
        Path(path).name.upper().startswith(("LICENSE", "COPYING"))
        for path in historical_files
    )
    historical_requirements_missing = "requirements.txt" not in historical_files
    gpt2_code_present = (gpt2_path / "src" / "model.py").is_file()
    model_assets_present = (
        (model_path / "hparams.json").is_file()
        and bool(checkpoint_assets)
    )
    entry_completed = entry_probe.get("completed") is True
    official_runnable = (
        entry_completed
        and gpt2_code_present
        and model_assets_present
    )
    comparison = evaluate_fair_comparison(
        {
            "same_prompts": False,
            "same_model": False,
            "same_payloads": False,
            "matched_controls": False,
            "recovery_measured": False,
            "same_evaluation": False,
            "official_end_to_end_runnable": official_runnable,
        }
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_policy": {
            "read_only_external_checkout": True,
            "network_access": False,
            "dependency_installation": False,
            "submodule_initialization": False,
            "model_downloads": False,
            "official_method_modified_or_vendored": False,
        },
        "provenance": {
            "repository": str(repository),
            "origin_url": origin_url or None,
            "historical_tag": historical_tag,
            "historical_tag_object": tag_object,
            "historical_tag_commit": tag_commit,
            "expected_historical_commit": expected_historical_commit,
            "head": head,
            "exact_tag": exact_tag or None,
            "historical_commit_metadata": _commit_metadata(repository, head),
            "current_ref": current_ref,
            "current_ref_commit": current_commit,
            "expected_current_commit": expected_current_commit,
            "current_commit_metadata": _commit_metadata(
                repository, current_ref
            ),
            "checks": provenance_checks,
            "historical_source_files": historical_hashes,
            "working_tree_status_before": status_before,
            "working_tree_status_after": status_after,
        },
        "historical_tag_tree": {
            "file_count": len(historical_files),
            "files": sorted(historical_files),
            "license_file_present": not historical_license_missing,
            "requirements_file_present": not historical_requirements_missing,
            "gpt2_gitlink": (
                gitlink_match.group(1) if gitlink_match else None
            ),
            "gpt2_submodule_status": submodule_status or None,
            "gpt2_submodule_initialized": submodule_initialized,
            "gpt2_source_model_py_present": gpt2_code_present,
            "gpt2_117m_model_assets_present": model_assets_present,
            "gpt2_117m_model_asset_files": model_asset_names,
            "gpt2_117m_checkpoint_files": checkpoint_assets,
        },
        "later_current_ref": {
            "files_added_or_changed_since_historical_tag": changed_names,
            "license_file_present": "LICENSE" in current_files,
            "license": license_facts,
            "requirements_file_present": "requirements.txt" in current_files,
            "selected_pinned_requirements": {
                name: requirements.get(name)
                for name in (
                    "tensorflow",
                    "numpy",
                    "torch",
                    "nltk",
                )
            },
            "requirements_sha256": (
                _sha256_text(later_requirements_text)
                if later_requirements_text
                else None
            ),
        },
        "licensing_interpretation": {
            "conclusion": (
                "The MIT license added on the later current branch likely covers "
                "the repository code distributed there, including code derived "
                "from the historical implementation. The acl-2019 tree itself "
                "omits the license file and notice."
            ),
            "historical_tag_archive_contains_license_notice": not historical_license_missing,
            "later_mit_license_is_positive_provenance_evidence": license_facts[
                "classification"
            ]
            == "MIT",
            "third_party_gpt2_code_and_model_assets_require_separate_provenance": True,
            "is_decisive_numeric_comparison_blocker": False,
            "caution": (
                "This is a repository-provenance interpretation, not a legal "
                "determination. Preserve the later MIT notice when permitted code "
                "is redistributed and audit third-party assets separately."
            ),
        },
        "method": method_evidence,
        "reproduction": {
            "standalone_huffman_dyadic_check": huffman_probe,
            "official_entry_point": entry_probe,
            "end_to_end_cover_generation_completed": entry_completed,
            "exact_payload_recovery_measured": False,
            "cover_quality_measured": False,
            "steganalysis_measured": False,
            "interpretation": (
                "The dyadic p check validates a small official Huffman/TV code "
                "path. It does not validate patient-Huffman generation with GPT-2, "
                "serialization/recovery, or any comparative outcome."
            ),
        },
        "dependency_and_asset_assessment": {
            "historical_tag_has_requirements": not historical_requirements_missing,
            "later_tensorflow_pin": requirements.get("tensorflow"),
            "later_requirements_are_2019_era": requirements.get("tensorflow")
            == "1.12.0",
            "tensorflow_available_to_entry_point": (
                None
                if entry_probe.get("skipped")
                else entry_completed
                or (
                    entry_probe.get("failure", {}).get("type")
                    != "ModuleNotFoundError"
                    or "tensorflow"
                    not in str(
                        entry_probe.get("failure", {}).get("message", "")
                    ).lower()
                )
            ),
            "gpt2_submodule_initialized": submodule_initialized,
            "gpt2_source_present": gpt2_code_present,
            "gpt2_117m_model_assets_present": model_assets_present,
            "decisive_reproducibility_blocker": not official_runnable,
        },
        "fair_comparison": comparison,
        "overall_conclusion": {
            "official_published_method_end_to_end_reproduced": official_runnable,
            "standalone_huffman_core_reproduced": huffman_probe.get("passed")
            is True,
            "numeric_patient_huffman_baseline_approved": False,
            "primary_rankcloak_experiments_blocked": False,
            "recommended_manuscript_treatment": (
                "Discuss Dai and Cai (ACL 2019) as a peer-reviewed prior method "
                "and report the read-only compatibility attempt transparently. "
                "Do not place a numeric patient-Huffman result in comparative "
                "figures or tables until all fair-comparison criteria are met."
            ),
        },
        "integrity": {
            "all_provenance_checks_passed": all(
                provenance_checks.values()
            ),
            "external_checkout_unchanged": (
                status_after == status_before and head_after == head
            ),
            "report_complete": True,
        },
    }
    return report


def write_report(
    report: Mapping[str, Any], output: Path, *, overwrite: bool = False
) -> None:
    output = Path(output)
    if output.exists() and not overwrite:
        raise ComparatorAuditError(
            f"Refusing to overwrite existing audit report: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pinned ACL 2019 patient-Huffman implementation without "
            "changing it, installing dependencies, or downloading model assets."
        )
    )
    parser.add_argument(
        "--repository", type=Path, default=DEFAULT_REPOSITORY
    )
    parser.add_argument(
        "--expected-historical-commit",
        default=DEFAULT_HISTORICAL_COMMIT,
    )
    parser.add_argument("--historical-tag", default=DEFAULT_HISTORICAL_TAG)
    parser.add_argument("--current-ref", default=DEFAULT_CURRENT_REF)
    parser.add_argument(
        "--expected-current-commit", default=DEFAULT_CURRENT_COMMIT
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Interpreter used only for local no-bytecode probes.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Collect provenance only; do not execute official Python files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file; stdout is always machine-readable JSON.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        report = audit_repository(
            args.repository,
            expected_historical_commit=args.expected_historical_commit,
            historical_tag=args.historical_tag,
            current_ref=args.current_ref,
            expected_current_commit=args.expected_current_commit,
            python_executable=args.python_executable,
            timeout_seconds=args.timeout_seconds,
            run_probes=not args.skip_probes,
        )
        if args.output:
            write_report(report, args.output, overwrite=args.overwrite)
    except ComparatorAuditError as exc:
        parser.exit(2, f"published comparator audit failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["integrity"]["all_provenance_checks_passed"]:
        return 3
    if not report["integrity"]["external_checkout_unchanged"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
