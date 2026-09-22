"""Offline, source-bound diagnostics for the bounded V4 first stage.

Historical records are read only. No generation or likelihood evaluation occurs.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path

from .reproducibility import sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def false_runs(eligible):
    """All maximal false runs as zero-based half-open intervals."""
    runs, start = [], None
    for i, value in enumerate(list(eligible) + [True]):
        if not value and start is None:
            start = i
        elif value and start is not None:
            runs.append((start, i))
            start = None
    return runs


def trace_diagnostics(record, window=32, low_progress=2):
    """Validate the retained trajectory and derive position and run summaries."""
    g = record["generation"]
    ids = g["embedding_token_ids"]
    n = len(ids)
    entropies = g["embedding_entropies_bits"]
    mask = g["embedding_eligible_mask"]
    roles = g["embedding_token_roles"]
    logp = g.get("embedding_log_probabilities", g.get("forced_log_probabilities"))
    ranks = g["embedding_observed_ranks"]
    pressure = g["embedding_rank_pressure_log_probability_gaps_nats"]
    if not n or any(len(x) != n for x in [entropies, mask, roles, logp, ranks, pressure]):
        raise ValueError("unaligned or empty position arrays")
    if not all(math.isfinite(float(v)) for a in [entropies, logp, pressure] for v in a):
        raise ValueError("nonfinite trace value")
    threshold = record["threshold_bits"]
    expected_mask = [threshold is None or h >= threshold for h in entropies]
    if mask != expected_mask:
        raise ValueError("stored eligibility disagrees with inclusive threshold")
    if roles != ["payload" if e else "ordinary_sampled_skip" for e in mask]:
        raise ValueError("stored roles disagree with eligibility")
    requested = int(g["requested_payload_rank_count"])
    consumed = sum(mask)
    if consumed != int(g["consumed_payload_rank_count"]):
        raise ValueError("consumed rank count disagrees with positions")
    if len(record["expected_ranks"]) != requested or consumed > requested:
        raise ValueError("requested rank count disagrees with representation")
    if bool(g["payload_completion"]) != (consumed == requested):
        raise ValueError("completion flag disagrees with consumed ranks")
    if g["consumed_payload_rank_indices"] != list(range(consumed)):
        raise ValueError("nonconsecutive consumed rank indices")
    expected = record["expected_ranks"]
    observed_payload = [r for r, e in zip(ranks, mask) if e]
    if observed_payload != expected[:consumed]:
        raise ValueError("payload ranks disagree with retained expected prefix")
    replay = record["saved_token_id_replay"]["replay"]
    if replay["embedding_eligible_mask"] != mask or replay["ranks"] != expected[:consumed]:
        raise ValueError("saved-ID replay disagrees with encoder")
    positions, cumulative, first_low = [], 0, None
    for i in range(n):
        before = cumulative
        cumulative += int(mask[i])
        start = max(0, i + 1 - window)
        rolling = sum(mask[start:i + 1])
        if i + 1 >= window and rolling <= low_progress and first_low is None:
            first_low = start
        positions.append({
            "plan_id": record["plan_id"], "position": i, "token_id": ids[i],
            "entropy_bits": entropies[i], "threshold_bits": threshold,
            "entropy_margin_bits": None if threshold is None else entropies[i] - threshold,
            "eligible": mask[i], "token_role": roles[i],
            "consumed_before": before, "consumed_after": cumulative,
            "remaining_after": requested - cumulative,
            "payload_rank": expected[before] if mask[i] else None,
            "observed_rank": ranks[i], "surprisal_nats": -logp[i],
            "rank_pressure_nats": pressure[i],
            "rolling_start": start, "rolling_position_count": i + 1 - start,
            "rolling_ranks_consumed": rolling,
            "rolling_eligible_fraction": rolling / (i + 1 - start),
            "rolling_window_full": i + 1 >= window,
        })
    runs = false_runs(mask)
    longest = max(runs, key=lambda pair: (pair[1] - pair[0], -pair[0]), default=(0, 0))
    row = record["plan_row"]
    summary = {k: row[k] for k in ["plan_id", "experimental_cell_id", "model_id", "payload_name", "payload_class", "representation_name", "prompt_template_id", "gate_level"]}
    summary.update({
        "requested_ranks": requested, "consumed_ranks": consumed,
        "remaining_ranks": requested - consumed, "tokens_used": n,
        "token_budget": record["fixed_payload"]["maximum_embedding_token_count"],
        "payload_completion": g["payload_completion"], "threshold_bits": threshold,
        "eligible_fraction": consumed / n,
        "mean_entropy_bits": sum(entropies) / n,
        "mean_entropy_margin_bits": None if threshold is None else sum(entropies) / n - threshold,
        "longest_below_start": longest[0], "longest_below_stop": longest[1],
        "longest_below_length": longest[1] - longest[0],
        "first_low_progress_window_start": first_low,
        "low_progress_window_definition": f"first full {window} positions with at most {low_progress} consumed ranks",
        "capacity_failure": g["capacity_failure"],
    })
    return summary, positions, [{"plan_id": record["plan_id"], "start": a, "stop": b, "length": b - a} for a, b in runs]


def select_comparators(records, failures, seed):
    """One identity-selected strict success per failed model/codec/prompt stratum."""
    def stratum(r):
        p = r["plan_row"]
        return tuple(p[k] for k in ["model_id", "representation_name", "prompt_template_id"])
    result = []
    for key in sorted({stratum(r) for r in failures}):
        candidates = [r for r in records if stratum(r) == key and r["plan_row"]["gate_level"] == "strict" and r["generation"]["payload_completion"]]
        if not candidates:
            raise ValueError("no successful strict comparator in failed stratum")
        result.append(min(candidates, key=lambda r: hashlib.sha256(f'{seed}|{r["plan_id"]}'.encode()).hexdigest()))
    return result


def aligned_windows(model, record, summary, flank=16, progress_window=32):
    """Use prefix detokenization byte offsets, never concatenated token pieces."""
    from .model_io import detokenize_bytes
    ids = record["generation"]["embedding_token_ids"]
    raw = detokenize_bytes(model, ids)
    if raw.decode("utf-8", errors="replace") != record["generation"]["embedding_text"]:
        raise ValueError("pinned tokenizer does not reproduce recorded rendering")
    offsets = [0]
    for i in range(1, len(ids) + 1):
        prefix = detokenize_bytes(model, ids[:i])
        if not raw.startswith(prefix) or len(prefix) < offsets[-1]:
            raise ValueError("non-prefix-stable detokenization requires separate alignment")
        offsets.append(len(prefix))
    a, b = summary["longest_below_start"], summary["longest_below_stop"]
    bounds = [("longest_below_threshold", a, b)] if b > a else [
        ("initial_window_no_below_threshold_run", 0, min(len(ids), progress_window))
    ]
    first = summary["first_low_progress_window_start"]
    if first is not None:
        bounds.append(("first_low_progress", first, min(len(ids), first + progress_window)))
    windows = []
    for kind, a, b in bounds:
        start, stop = max(0, a - flank), min(len(ids), b + flank)
        bs, be = offsets[start], offsets[stop]
        # A tokenizer boundary can divide a multi-byte code point. Expand only
        # the display slice. The exact event token and byte bounds remain saved.
        while bs and raw[bs] & 0xC0 == 0x80:
            bs -= 1
        while be < len(raw) and raw[be] & 0xC0 == 0x80:
            be += 1
        windows.append({"plan_id": record["plan_id"], "kind": kind,
            "event_start": a, "event_stop": b, "event_byte_start": offsets[a],
            "event_byte_stop": offsets[b], "window_token_start": start,
            "window_token_stop": stop, "display_byte_start": bs, "display_byte_stop": be,
            "text": raw[bs:be].decode("utf-8", errors="replace"),
            "alignment_status": "verified_pinned_prefix_bytes",
            "utf8_replacement_in_display": "\ufffd" in raw[bs:be].decode("utf-8", errors="replace"),
            "interpretation": "descriptive window, no linguistic causal attribution"})
    return offsets, windows


def export_filter(root, out):
    """Extract the literal list from the predicate AST, preserving order."""
    path = root / "rankcloak/token_filters.py"
    tree = ast.parse(path.read_text())
    predicate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_safe_text_token_piece")
    assignment = next(n for n in ast.walk(predicate) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "blocked_substrings" for t in n.targets))
    fragments = ast.literal_eval(assignment.value)
    rules = [
        ("empty", "Reject an empty decoded piece."),
        ("replacement", "Reject any occurrence of Unicode U+FFFD."),
        ("ascii_control", "Reject any code point below 32 except LF (10) and TAB (9). CR (13) is rejected. DEL (127) and other Unicode controls are not rejected by this rule alone."),
    ]
    rows = [{"rule_id": k, "criterion": v, "literal_json": ""} for k, v in rules]
    rows += [{"rule_id": f"substring_{i:02d}", "criterion": "Reject occurrence in piece.lower().", "literal_json": json.dumps(v)} for i, v in enumerate(fragments, 1)]
    rows += [
        {"rule_id": "hash_start", "criterion": "Reject piece.strip().startswith('#') or startswith('##'). The second condition is redundant. No general whitespace exclusion.", "literal_json": ""},
        {"rule_id": "backslashes", "criterion": "Reject two consecutive backslashes or at least two backslashes anywhere in the original piece.", "literal_json": ""},
    ]
    rows += [{"rule_id": key, "criterion": value, "literal_json": ""} for key, value in [
        ("individual_decoding", "Decode each vocabulary token independently with safe_detokenize. Byte output is decoded as UTF-8 with replacement."),
        ("decoding_exceptions", "A detokenization exception produces an angle-bracket marker rejected by the predicate. An exception reaching mask construction produces an empty rejected piece."),
        ("mask_cache", "Cache the Boolean vocabulary mask by model object identity and filter name."),
        ("unmasked", "None, an empty filter name, and the literal name none return no mask."),
        ("mask_errors", "Unknown filter name, unavailable vocabulary size, all-rejected vocabulary, and incompatible mask length raise errors."),
        ("selection", "Convert scores to float64 and fully sort allowed token IDs by descending score, then ascending token ID. Rank is positive and one-indexed. Out-of-range requested ranks raise errors."),
        ("recovery", "Count strictly higher allowed scores and equal scores at lower allowed token IDs, then add one. Excluded or out-of-range target IDs raise errors."),
        ("score_domain", "The historical helpers do not explicitly reject nonfinite logits. The stated inverse ordering contract assumes finite scores."),
        ("special_tokens", "Safe-text filtering has no implicit special-token-ID exclusion. Ordinary sampling separately excludes exposed BOS, EOS, and EOT IDs."),
        ("roundtrip_filter", "The isolated roundtrip filter is a distinct additional tokenization test, not part of safe_text_filter_v1."),
    ]]
    write_csv(out / "source_tables/filter_rules.csv", rows)
    write_json(out / "provenance/filter_contract.json", {
        "filter_name": "safe_text_filter_v1", "source_path": str(path.relative_to(root)),
        "source_sha256": sha256_file(path), "predicate_line": predicate.lineno,
        "blocked_substrings": fragments,
        "related_sources": {p: sha256_file(root / p) for p in ["rankcloak/model_io.py", "rankcloak/rank_codec.py", "rankcloak/revision_protocol.py", "rankcloak/revision_runner.py"]},
        "selection": "np.lexsort((token_ids, -scores[token_ids])); full allowed-set sort",
        "rank_recovery": "count strictly higher scores and equal scores at lower token IDs",
        "historical_behavior_modified": False,
    })
    # Each literal is separately represented so LaTeX treats backslashes as data.
    cells = ", ".join(r"\texttt{\detokenize{" + s + "}}" if "{" not in s else r"\texttt{\{\textbackslash}" for s in fragments)
    tex = r"""\subsection*{Complete deterministic token filter specification}\label{sec:v4-filter}
\textbf{V4 working draft.} This specification documents the existing filter without changing its behavior.
For \code{safe_text_filter_v1}, the following rules apply to each individually decoded vocabulary token.
\begin{table}[H]
\centering\small
\begin{tabularx}{\textwidth}{L{0.20\textwidth}Y}
\toprule
Rule & Exact exclusion criterion \\
\midrule
Empty or replacement & Empty string or any U+FFFD replacement character. \\
Controls & Any code point below 32 except newline and tab. Carriage return is excluded. DEL and other Unicode controls are not excluded by this rule alone. \\
Literal fragments & Any case insensitive occurrence of FRAGMENTS. \\
Start rule & After stripping whitespace, a piece starting with \texttt{\#} or \texttt{\#\#}. The second check is redundant. \\
Backslashes & Two consecutive backslashes or at least two backslashes anywhere in the piece. \\
\bottomrule
\end{tabularx}
\caption{Complete exclusion rules of the retained safe-text filter. A piece is allowed only if none applies.}\label{tab:v4-filter}
\end{table}
The filter decodes one token at a time with \code{safe_detokenize}. Byte output uses UTF-8 replacement decoding. If detokenization raises an exception, the wrapper returns an angle-bracket token marker, which the filter rejects. Any exception that reaches mask construction instead yields an empty piece, which is also rejected. Whitespace alone is allowed unless it contains an excluded control. Filtering does not guarantee that concatenated pieces are free of markup or tokenization changes.

The mask is cached by model object identity and filter name. No filter, the empty name, and \code{none} leave the vocabulary unmasked. Unknown names, missing vocabulary size, an entirely rejected vocabulary, incompatible mask length, unavailable requested ranks, and a target excluded by the mask raise errors. The selector validates positive one-indexed ranks. Masks are constructed with the model vocabulary length.

Candidate scores are converted to float64. The stated inverse ordering contract assumes finite logits. The historical helpers do not explicitly reject nonfinite scores. Selection sorts all allowed candidates by descending logit and ascending token ID. Recovery of a known token counts greater logits and equal logits at lower IDs. Both operations use the same allowed set. The isolated roundtrip filter adds a separate tokenization test. Ordinary sampling separately excludes BOS, EOS, and EOT identifiers where exposed by the backend. These exclusions are not implicit safe-text rules.

The \href{https://github.com/mantzaris/llm-rankcloak/blob/60d23fffe5962d4e0b652fa045d762ca10bc3dcc/rankcloak/token_filters.py}{versioned token-filter source} is available in the repository. Its exact version and SHA-256 are retained in the V4 filter audit.
""".replace("FRAGMENTS", cells)
    (out / "manuscript_tables").mkdir(exist_ok=True)
    (out / "manuscript_tables/filter_methods.tex").write_text(tex)
    return rows


def run(root, out, align=False):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "configs/revision_v4/stage1_offline.json"
    cfg = json.loads(cfg_path.read_text())
    source = root / cfg["entropy_source_table"]
    table = list(csv.DictReader(source.open()))
    if len(table) != 720 or len({r['plan_id'] for r in table}) != 720:
        raise ValueError("unexpected source denominator or duplicate identity")
    paths = sorted((root / cfg["entropy_raw_root"]).rglob("entropy_trial__*.json"))
    all_records = [json.loads(p.read_text()) for p in paths]
    if {r['plan_id'] for r in all_records} != {r['plan_id'] for r in table} or len(all_records) != 720:
        raise ValueError('raw ledger identities disagree with the complete source table')
    records = [r for r in all_records if r['record_type'] == 'entropy_rankcloak_trial']
    by_id = {r["plan_id"]: r for r in records}
    path_by_id = {r["plan_id"]: p for r, p in zip(all_records, paths)}
    if len(by_id) != 360:
        raise ValueError("expected 360 distinct historical RankCloak attempts")
    denominators = []
    summaries = {}
    for r in records:
        summary, _, _ = trace_diagnostics(r, cfg['rolling_window_positions'], cfg['low_progress_maximum_ranks_per_full_window'])
        summaries[r['plan_id']] = summary
    for level in ['ungated', 'moderate', 'strict']:
        group = [r for r in records if r['plan_row']['gate_level'] == level]
        rows = [r for r in table if r['gate_level'] == level]
        if len(group) != 120 or len(rows) != 240:
            raise ValueError("gate-level denominators changed")
        denominators.append({'gate_level': level, 'rankcloak_attempts': 120, 'ordinary_controls': 120,
            'completed_payloads': sum(r['generation']['payload_completion'] for r in group)})
    for row in table:
        if row['population'] != 'rankcloak':
            continue
        d = by_id[row['plan_id']]
        s = summaries[row['plan_id']]
        for key in ['model_id', 'payload_name', 'representation_name', 'prompt_template_id', 'gate_level', 'experimental_cell_id']:
            if row[key] != s[key]:
                raise ValueError(f'source table identity mismatch {key}')
        if (row['payload_completion'] == 'True') != s['payload_completion'] or int(row['generated_token_count']) != s['tokens_used'] or int(float(row['payload_rank_count'])) != s['requested_ranks']:
            raise ValueError('source table outcome mismatch')
        if not math.isclose(float(row['eligible_position_fraction']), s['eligible_fraction'], abs_tol=1e-12):
            raise ValueError('source table eligibility mismatch')
    failures = sorted([r for r in records if not r['generation']['payload_completion']], key=lambda r:r['plan_id'])
    if {r['plan_id'].split('__')[-1] for r in failures} != set(cfg['expected_failure_suffixes']):
        raise ValueError('failure identities changed')
    for r in failures:
        s = summaries[r['plan_id']]
        if [s['requested_ranks'], s['consumed_ranks'], s['tokens_used']] != cfg['expected_failure_suffixes'][r['plan_id'].split('__')[-1]]:
            raise ValueError('failure accounting changed')
        if s['gate_level'] != 'strict' or s['representation_name'] != 'ascii_b16' or s['tokens_used'] != s['token_budget'] or s['token_budget'] != 6*s['requested_ranks']:
            raise ValueError('failure budget contract changed')
    selected = {r['plan_id']: 'strict_failure' for r in failures}
    for failure in failures:
        cell = failure['plan_row']['experimental_cell_id']
        for level in cfg['matched_levels']:
            matches = [r for r in records if r['plan_row']['experimental_cell_id'] == cell and r['plan_row']['gate_level'] == level]
            if len(matches) != 1:
                raise ValueError('matched gate record is missing or nonunique')
            selected[matches[0]['plan_id']] = 'matched_' + level
    for r in select_comparators(records, failures, cfg['seed']):
        selected[r['plan_id']] = 'identity_selected_strict_success'
    position_rows, run_rows, summary_rows = [], [], []
    for identity, reason in sorted(selected.items()):
        s, positions, runs = trace_diagnostics(by_id[identity], cfg['rolling_window_positions'], cfg['low_progress_maximum_ranks_per_full_window'])
        p = path_by_id[identity]
        s.update({'selection_role': reason, 'source_path': str(p.relative_to(root)), 'source_sha256': sha256_file(p)})
        summary_rows.append(s)
        position_rows.extend(positions)
        run_rows.extend(runs)
    windows, alignment = [], []
    if align:
        from .revision_tokenizer_preflight import load_vocab_only_tokenizer
        requirements = json.loads((root / 'configs/revision_v3/generation_requirements.json').read_text())
        import importlib.metadata
        if importlib.metadata.version('llama-cpp-python') != requirements['required_backend']['version']:
            raise ValueError('tokenizer backend does not match pinned version')
        for model_id in sorted({r['model_id'] for r in summary_rows}):
            artifact = next(a for a in requirements['artifacts'] if a['model_id'] == model_id)
            p = root / artifact['expected_path']
            if not p.exists():
                alignment.append({'model_id':model_id, 'status':'pending_missing_pinned_file', 'path':artifact['expected_path']})
                continue
            if p.stat().st_size != artifact['size_bytes'] or sha256_file(p) != artifact['sha256']:
                raise ValueError('local tokenizer artifact differs from pinned bytes')
            model = load_vocab_only_tokenizer(p)
            try:
                for summary in [s for s in summary_rows if s['model_id'] == model_id]:
                    identity = summary['plan_id']
                    offsets, excerpts = aligned_windows(model, by_id[identity], summary, cfg['text_flank_tokens'], cfg['rolling_window_positions'])
                    windows.extend(excerpts)
                    for row in position_rows:
                        if row['plan_id'] == identity:
                            i = row['position']
                            row.update({'byte_start':offsets[i], 'byte_stop':offsets[i+1]})
            finally:
                model.close()
            alignment.append({'model_id':model_id, 'status':'verified_pinned_prefix_bytes', 'artifact_sha256':artifact['sha256'], 'backend_version':'0.3.23', 'inference_executed':False})
    write_csv(out / 'source_tables/entropy_six_failures.csv', [s for s in summary_rows if s['selection_role']=='strict_failure'])
    write_csv(out / 'source_tables/entropy_selected_runs.csv', summary_rows)
    write_csv(out / 'source_tables/entropy_positions.csv', position_rows)
    write_csv(out / 'source_tables/entropy_below_threshold_runs.csv', run_rows)
    write_csv(out / 'source_tables/entropy_denominators.csv', denominators)
    write_csv(out / 'source_tables/entropy_text_windows.csv', windows)
    export_filter(root, out)
    inputs = [cfg_path, source, root/'rankcloak/revision_v4_stage1.py', root/'scripts/analyze_revision_v4_stage1.py', root/'rankcloak/revision_tokenizer_preflight.py', root/'configs/revision_v3/entropy_gate.json', root/'configs/revision_v3/generation_requirements.json', root/'rankcloak/revision_v3_entropy.py', root/'rankcloak/revision_v3_diagnostics.py'] + paths
    report = ['# Stage 1 entropy diagnostics', '', 'Exploratory analysis of retained trajectories. No historical generation was repeated.', '',
        'All 360 RankCloak records reconcile with the 720-row table, which also contains 360 ordinary controls. Each gate retains 120 attempts. Ungated and moderate complete 120 each. Strict completes 114 and retains all six failures.', '',
        f'{len(selected)} trajectories are selected. Twelve are the exact ungated and moderate matches to the six failures. Three strict successes are selected by the frozen hash rule within failed model, representation, and prompt strata. They are descriptive comparisons with different payloads.', '',
        '| Trial suffix | Requested | Consumed | Positions | Longest below-threshold run | Eligible fraction |', '| --- | --- | --- | --- | --- | --- |']
    for s in summary_rows:
        if s['selection_role']=='strict_failure':
            report.append(f"| {s['plan_id'].split('__')[-1]} | {s['requested_ranks']} | {s['consumed_ranks']} | {s['tokens_used']} | {s['longest_below_length']} | {s['eligible_fraction']:.6f} |")
    report += ['', 'Every failed trajectory exhausts its 6L budget with fewer than L eligible positions. Thus its realized eligible fraction is below 1/6. This is exact capacity accounting. It does not establish that a linguistic pattern caused the low entropy.', '',
        'Position tables retain entropy, inclusive eligibility, token role, observed rank, surprisal, rank pressure, cumulative consumption, remaining ranks, and trailing 32-position progress. Partial initial windows are explicitly labeled. The first low-progress window is the first full window with at most two consumed ranks. All maximal below-threshold runs are retained and longest-run ties choose the earliest.', '',
        'The window table uses pinned vocabulary-only prefix detokenization and byte offsets when alignment is verified. It never concatenates isolated token pieces. Event bounds remain exact while display bounds can expand to avoid cutting UTF-8. Linguistic interpretations remain exploratory and must be compared with matched and successful traces.', '',
        'The ordinary-development 75th percentile does not imply 25 percent eligibility along a forced generation trajectory. Different contexts and sampled skips change that trajectory. No causal claim follows from the selected examples.', '',
        'Reproduce with `PYTHONPATH=. .venv-generation-v3/bin/python scripts/analyze_revision_v4_stage1.py --align-tokenizers`. The manifest binds input bytes and selection rules. Figure generation is a separate CPU-only command.']
    (out/'ENTROPY_FAILURE_REPORT.md').write_text('\n'.join(report)+'\n')
    generated = [
        'source_tables/entropy_six_failures.csv', 'source_tables/entropy_selected_runs.csv',
        'source_tables/entropy_positions.csv', 'source_tables/entropy_below_threshold_runs.csv',
        'source_tables/entropy_denominators.csv', 'source_tables/entropy_text_windows.csv',
        'source_tables/filter_rules.csv', 'manuscript_tables/filter_methods.tex',
        'provenance/filter_contract.json', 'ENTROPY_FAILURE_REPORT.md',
    ]
    inputs += [root / p for p in ['rankcloak/token_filters.py', 'rankcloak/model_io.py',
        'rankcloak/rank_codec.py', 'rankcloak/revision_protocol.py', 'rankcloak/revision_runner.py']]
    write_json(out / 'provenance/offline_manifest.json', {'schema_version':'rankcloak-v4-stage1-offline-v1',
        'evidence_status':cfg['evidence_status'], 'selection_rules':cfg, 'denominators':denominators,
        'selected_records':selected, 'alignment':alignment, 'model_generation_performed':False,
        'inputs':{str(p.relative_to(root)):sha256_file(p) for p in inputs},
        'outputs':{p:sha256_file(out / p) for p in generated},
    })
    return {'selected_runs':len(selected), 'positions':len(position_rows), 'failures':len(failures), 'alignment':alignment}
