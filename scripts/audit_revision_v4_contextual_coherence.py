"""Independent source joins and post-scoring quotation mapping, without inference."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import numpy as np
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT, CONFIG, judge_content
from rankcloak.revision_v4_stage2_common import read_json, read_jsonl, file_hash, digest, atomic_json, write_jsonl


def audit_sources():
    plan = OUT / 'plans/final_sample'
    freeze = read_json(plan / 'freeze.json')
    selection = read_json(plan / 'selection.json')
    for name, expected in freeze['files'].items():
        assert file_hash(plan / name) == expected, name
    for name, expected in freeze['source_hashes'].items():
        assert file_hash(ROOT / name) == expected, name
    messages = read_jsonl(plan / 'messages.jsonl')
    source_records = {}
    for name, expected in selection['source_hashes'].items():
        assert file_hash(ROOT / name) == expected, name
        for row in read_jsonl(ROOT / name):
            if row.get('record_type') == 'rankcloak_trial':
                source_records[row['trial_id']] = row
    controls = {r['request_id']: r['request'] for r in read_jsonl(plan / 'controls.jsonl')}
    old = {r['boundary_id']: r for r in read_jsonl(ROOT / 'results/revision_v4/stage2/plans/amendment1/boundaries.jsonl')}
    counts = Counter()
    boundary_positions = []
    for message in messages:
        record = source_records[message['trial_id']]
        segment = record['segments'][message['segment_index']]
        assert digest(record) == message['source_record_sha256']
        assert segment['full_text'] == message['text']
        assert segment['full_token_ids'] == message['full_token_ids']
        assert segment['prompt'] == message['prompt']
        control = controls[message['control_request_id']]
        assert control['context_ids'] == segment['context_token_ids']
        assert control['prompt'] == segment['prompt']['prompt_text']
        assert control['generator'] == message['generator']
        assert control['filter'] == record['token_filter'] == 'safe_text_filter_v1'
        assert control['tail_policy'] == record['tail_policy'] == 'dynamic_completion_v1'
        assert control['total_budget'] == 264 and message['forced_stop'] == 8
        previous = old[message['message_id']]
        counts[(message['generator'], 'previously_eligible' if previous['eligible'] else 'previously_ineligible')] += 1
        for reason in previous['exclusion_reasons']:
            counts[(message['generator'], reason)] += 1
        forced = segment['forced_text']
        # A decoded prefix is not assumed to be a literal character prefix.
        aligned = message['text'].startswith(forced)
        boundary_positions.append({'message_id': message['message_id'], 'trial_id': message['trial_id'],
            'segment_index': message['segment_index'], 'forced_tokens': message['forced_stop'],
            'character_position': len(forced) if aligned else None,
            'byte_position': len(forced.encode('utf-8')) if aligned else None,
            'literal_decoded_prefix_alignment': aligned})
    requests = {r['request_id']: r['request'] for r in read_jsonl(plan / 'judge_requests.jsonl')}
    units = read_jsonl(plan / 'unit_requests.jsonl')
    generated_controls = {}
    for path in (OUT / 'raw/controls').glob('*.jsonl'):
        old_mask_path = ROOT / 'results/revision_v1/primary_v2' / path.stem / 'filter_masks' / (path.stem+'__safe_text_filter_v1.json')
        old_mask = read_json(old_mask_path)
        allowed = set(old_mask['allowed_token_ids'])
        expected_mask_digest = digest([int(i in allowed) for i in range(old_mask['vocabulary_size'])])
        for row in read_jsonl(path):
            assert row['result']['filter_sha256'] == expected_mask_digest
            assert row['result']['allowed_token_count'] == old_mask['allowed_token_count']
            generated_controls[row['request_id']] = row['result']['generation']
    message_map = {m['message_id']: m for m in messages}
    for unit in units:
        message = message_map[unit['message_id']]
        request = requests[unit['request_id']]
        text = message['text'] if unit['arm'] == 'encoded' else generated_controls[unit['control_request_id']]['full_text']
        assert request['message'] == text
        assert request['context'] == message['prompt']['prompt_text']
        rendered = judge_content(request['context'], request['message'])
        # Labels exist in the private unit map but never in the supplied item.
        supplied = json.loads(rendered.split('\n\nEVALUATION DATA\n', 1)[1])
        assert set(supplied) == {'anonymous_item', 'context', 'message'}
        assert supplied['message'] == text
        assert supplied['context'] == message['prompt']['prompt_text']
    atomic_json(OUT / 'validation/source_audit.json', {
        'faithful_complete_text_checks': len(messages), 'matched_control_contract_checks': len(messages),
        'blinded_logical_unit_checks': len(units), 'frozen_source_hash_checks': len(freeze['source_hashes']),
        'new_missing_text': 0, 'previously_ineligible_messages_retained': sum(n for (g,r),n in counts.items() if r == 'previously_ineligible'),
        'old_eligibility_counts': [{'generator': g, 'reason': r, 'count': n} for (g,r),n in sorted(counts.items())],
        'boundary_character_alignment_available': sum(r['literal_decoded_prefix_alignment'] for r in boundary_positions),
        'interpretation': 'Old joint window eligibility does not restrict the new intact-message sample. Reasons may overlap.'})
    write_jsonl(OUT / 'validation/boundary_positions.jsonl', boundary_positions)
    return boundary_positions


def audit_scores(boundary_positions):
    boundaries = {r['message_id']: r for r in boundary_positions}
    joined = read_jsonl(OUT / 'analysis/joined_scores.jsonl')
    quote_rows = []
    for row in joined:
        if row['arm'] != 'encoded':
            continue
        boundary = boundaries[row['message_id']]['character_position']
        quote = row.get('quote', '')
        starts = row.get('quote_character_starts', [])
        spans = [{'start': start, 'stop': start + len(quote),
                  'crosses_boundary': start < boundary < start + len(quote) if boundary is not None else None,
                  'distance_to_boundary': min(abs(start-boundary), abs(start+len(quote)-boundary)) if boundary is not None else None}
                 for start in starts]
        quote_rows.append({'message_id': row['message_id'], 'judge_model_id': row['judge_model_id'],
            'disruption': row.get('disruption'), 'parse_status': row['parse_status'], 'quote': quote, 'quote_status': row.get('quote_status'),
            'literal_matches': len(starts), 'boundary_character_position': boundary, 'spans': spans})
    write_jsonl(OUT / 'analysis/disruption_quote_alignment.jsonl', quote_rows)
    # Independently compute point estimates with explicit three-level arithmetic.
    values = defaultdict(lambda: defaultdict(list))
    scores = defaultdict(dict)
    for row in joined:
        scores[(row['message_id'], row['arm'])][row['judge_model_id']] = row
    messages = read_jsonl(OUT / 'plans/final_sample/messages.jsonl')
    for message in messages:
        arms = [scores[(message['message_id'], arm)] for arm in ['encoded', 'ordinary']]
        if all(len(a) == 2 and all(r['parse_status'] == 'valid' for r in a.values()) for a in arms):
            for arm, rows in zip(['encoded', 'ordinary'], arms):
                mean = sum(r['disruption'] <= 1 for r in rows.values()) / 2
                values[arm][(message['payload_name'], message['trial_id'])].append(mean)
    point_estimates = {}
    payload_estimates = {}
    for arm, trials in values.items():
        payloads = defaultdict(list)
        for (payload, trial), numbers in trials.items():
            payloads[payload].append(sum(numbers) / len(numbers))
        payload_estimates[arm] = {p:sum(v)/len(v) for p,v in payloads.items()}
        means = list(payload_estimates[arm].values())
        point_estimates[arm] = sum(means) / len(means)
    point_estimates['difference'] = point_estimates['encoded'] - point_estimates['ordinary']
    summary = read_json(OUT / 'analysis/summary.json')
    primary = next(r for r in summary['summaries'] if r['group'] == 'panel')
    for arm, value in point_estimates.items():
        assert abs(value - primary[arm]['estimate']) < 1e-12
    classes = {m['payload_name']:m['payload_class'] for m in messages}
    strata = {c:sorted(p for p in classes if classes[p] == c) for c in sorted(set(classes.values()))}
    rng = np.random.default_rng(read_json(CONFIG)['analysis']['bootstrap_seed'])
    independent_draws = {field:[] for field in ['encoded','ordinary','difference']}
    for _ in range(2000):
        selected = [str(p) for names in strata.values() for p in rng.choice(names, len(names), replace=True)]
        arm_means = {}
        for arm in ['encoded','ordinary']:
            available = [payload_estimates[arm][p] for p in selected if p in payload_estimates[arm]]
            arm_means[arm] = sum(available)/len(available)
            independent_draws[arm].append(arm_means[arm])
        independent_draws['difference'].append(arm_means['encoded']-arm_means['ordinary'])
    independent_intervals = {}
    for field, estimates in independent_draws.items():
        lower,upper = np.quantile(estimates,[.025,.975])
        assert abs(lower-primary[field]['lower']) < 1e-12
        assert abs(upper-primary[field]['upper']) < 1e-12
        independent_intervals[field] = [float(lower),float(upper)]
    valid = [r for r in quote_rows if r['parse_status'] == 'valid']
    disrupted = [r for r in valid if r['disruption'] >= 2]
    atomic_json(OUT / 'validation/score_audit.json', {'independently_recomputed_point_estimates': point_estimates,
        'independently_recomputed_bootstrap_intervals': independent_intervals,
        'encoded_judge_slots': len(quote_rows), 'disrupted_encoded_judge_slots': len(disrupted),
        'disrupted_slots_with_literal_quote': sum(bool(r['literal_matches']) for r in disrupted),
        'disrupted_slots_with_boundary_crossing_quote': sum(any(s['crosses_boundary'] for s in r['spans']) for r in disrupted),
        'disrupted_slots_with_multiple_matches': sum(r['literal_matches'] > 1 for r in disrupted),
        'mapping_rule': 'Post-scoring non-overlapping literal substring matching, all recorded occurrences retained. A boundary crossing is strict interior containment, not proximity or causal attribution.',
        'limitations': 'Judges never received boundaries. Missing or nonliteral quotations do not establish absence of a defect. Repeated exact cached requests retain all inferential mappings.'})


def audit_missingness():
    joined = read_jsonl(OUT/'analysis/joined_scores.jsonl')
    panels = read_jsonl(OUT/'analysis/panel_message_scores.jsonl')
    messages = {r['message_id']:r for r in read_jsonl(OUT/'plans/final_sample/messages.jsonl')}
    invalid = [r for r in joined if r['parse_status'] != 'valid']
    strata = []
    for generator in sorted({r['generator'] for r in panels}):
        for schedule in ['all', *sorted({r['schedule'] for r in panels})]:
            selected = [r for r in panels if r['generator'] == generator and (schedule == 'all' or r['schedule'] == schedule)]
            complete = [r for r in selected if r['difference'] is not None]
            strata.append({'generator':generator, 'schedule':schedule, 'selected_messages':len(selected), 'complete_messages':len(complete),
                'selected_trials':len({r['trial_id'] for r in selected}), 'complete_trials':len({r['trial_id'] for r in complete}),
                'selected_payloads':len({r['payload_name'] for r in selected}), 'complete_payloads':len({r['payload_name'] for r in complete})})
    lengths = []
    for valid in [True, False]:
        selected = [messages[r['message_id']] for r in panels if (r['difference'] is not None) == valid]
        lengths.append({'pair_complete':valid, 'pairs':len(selected),
            'encoded_mean_tokens':float(np.mean([len(r['full_token_ids']) for r in selected])) if selected else None,
            'encoded_median_tokens':float(np.median([len(r['full_token_ids']) for r in selected])) if selected else None})
    disagreement = {}
    for arm in ['encoded', 'ordinary']:
        values = [r[arm+'_disagreement'] for r in panels if r[arm+'_disagreement'] is not None]
        disagreement[arm] = {'both_judges_valid_messages':len(values), 'binary_disagreements_unweighted_count':int(sum(values))}
    report = {'invalid_unique_responses':len({r['request_id'] for r in invalid}), 'invalid_logical_slots':len(invalid),
        'incomplete_pairs':sum(r['difference'] is None for r in panels), 'complete_pairs':sum(r['difference'] is not None for r in panels),
        'complete_trials':len({r['trial_id'] for r in panels if r['difference'] is not None}),
        'missing_by_judge_arm':[{'judge':key[0], 'arm':key[1], 'logical_slots':count,
            'unique_requests':len({r['request_id'] for r in invalid if r['judge_model_id']==key[0] and r['arm']==key[1]})}
            for key,count in sorted(Counter((r['judge_model_id'],r['arm']) for r in invalid).items())],
        'missing_by_generator_class':[{'generator':key[0], 'class':key[1], 'arm':key[2], 'logical_slots':count}
            for key,count in sorted(Counter((r['generator'],r['payload_class'],r['arm']) for r in invalid).items())],
        'complete_population_by_stratum':strata, 'encoded_lengths_by_pair_availability':lengths,
        'disagreement_availability':disagreement,
        'interpretation':'Complete-pair estimates condition on availability. Missingness includes reused controls and longer encoded messages. All-slot bounds retain the full selected population.'}
    atomic_json(OUT/'analysis/missingness_details.json',report)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--sources-only', action='store_true'); args=parser.parse_args()
    positions = audit_sources()
    if not args.sources_only:
        audit_scores(positions)
        audit_missingness()
    print('Source audit passed' if args.sources_only else 'Source and independent score audit passed')


if __name__ == '__main__':
    main()
