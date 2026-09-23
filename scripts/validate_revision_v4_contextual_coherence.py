"""Strict current-study scientific, quotation and preservation checks, without inference."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
from rankcloak.revision_v4_contextual_coherence import ROOT, OUT, CONFIG, parse_score, judge_content
from rankcloak.revision_v4_stage2_common import read_json, read_jsonl, file_hash, digest, atomic_json
from scripts.prepare_revision_v4_stage1 import escape
from scripts.validate_revision_v4_stage2 import verify_quotes

ALLOWED_EXISTING_CHANGES={
    'paperV4/scientific_reports/'+name for name in [
        'main4.tex','main4.pdf','main4.bbl','main4_submission.tex','supplementary4.tex','supplementary4.pdf','supplementary4.bbl',
        'references.bib','v4_stage2_methods.tex','v4_stage2_results.tex','v4_stage2_supplement.tex']
} | {'.gitignore','README.md','paperV4/response/response_to_reviewers_v4.tex','paperV4/response/response_to_reviewers_v4.pdf',
    'paperV4/cover_letter/cover_letter_v4.tex','paperV4/cover_letter/cover_letter_v4.pdf',
    'paperV4/REVIEW_RESPONSE_MATRIX.md','paperV4/ACTIONS_BEFORE_SUBMISSION.md'}


def main():
    initial=read_json(OUT/'provenance/initial_state.json');changed=[]
    for name,expected in initial['protected_git_blobs'].items():
        if name in ALLOWED_EXISTING_CHANGES:continue
        path=ROOT/name
        if not path.is_file():changed.append(name);continue
        raw=path.read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual!=expected:changed.append(name)
    if changed:raise ValueError('Protected historical or unrelated files changed: '+str(changed))
    plan=OUT/'plans/final_sample';freeze=read_json(plan/'freeze.json')
    for name,expected in freeze['files'].items():assert file_hash(plan/name)==expected,name
    for name,expected in freeze['source_hashes'].items():assert file_hash(ROOT/name)==expected,name
    for name,expected in freeze['control_cache_hashes'].items():assert file_hash(ROOT/name)==expected,name
    assert freeze['rankcloak_judge_outcomes_seen'] is False and freeze['rubric_amendments']==0
    summary=read_json(OUT/'analysis/summary.json')
    assert summary['status']=='complete'
    assert (summary['selected_payloads'],summary['trials'],summary['encoded_messages'],summary['logical_judge_units'],summary['unique_judge_requests'])==(48,288,576,2304,1182)
    requests={r['request_id']:r['request'] for r in read_jsonl(plan/'judge_requests.jsonl')}
    contract=digest(read_json(OUT/'provenance/inference_contract.json'));seen=set();attempts=0;actual_tokens=Counter()
    for path in (OUT/'raw/judging').glob('*.jsonl'):
        for row in read_jsonl(path):
            identity=row['request_id'];assert identity not in seen;seen.add(identity)
            assert requests[identity]==row['request'] and digest(row['request'])==identity
            assert row['contract_sha256']==contract
            assert row['result']['judge_input']==judge_content(row['request']['context'],row['request']['message'])
            assert row['result']['rendered_prompt_sha256']==digest(row['result']['rendered_prompt'])
            attempt_rows=row['result']['attempts'];assert 1<=len(attempt_rows)<=2
            attempts+=len(attempt_rows)
            for attempt in attempt_rows:
                response=attempt['response'];choice=response['choices'][0]
                assert response['usage']['prompt_tokens']==freeze['actual_input_token_counts'][identity]
                assert attempt['cap'] in [128,192]
                parsed=parse_score(choice['message']['content'],row['request']['message'])
                if choice['finish_reason']=='length':assert attempt['parsed']['parse_status']=='invalid'
                else:assert parsed==attempt['parsed']
                actual_tokens[path.stem+'_input']+=response['usage']['prompt_tokens']
                actual_tokens[path.stem+'_output']+=response['usage']['completion_tokens']
            if len(attempt_rows)==2:assert attempt_rows[0]['parsed']['parse_status']=='invalid'
            assert row['result']['parsed']==attempt_rows[-1]['parsed']
    assert seen==set(requests)
    response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text()
    blocks=verify_quotes((ROOT/'paperV4/response/requests.txt').read_text(),read_json(ROOT/'paperV4/response/review_comments.json'),response)
    status=read_json(OUT/'response_status.json');assert status['written_complete']==16 and len(status['response_sets'])==16
    assert status['empirical_R1_4']=='automated_contextual_evidence_supplied_human_perception_unmeasured'
    assert status['public_current_study_coverage'] is False and status['journal_submission_performed'] is False
    locations=read_json(OUT/'validation/locations.json')
    for row in status['response_sets']:
        pattern=r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id'])
        bodies=re.findall(pattern,response,re.S);assert len(bodies)==1
        for heading,key in [('Answer','direct_answer'),('Change and evidence','evidence_change'),('Location','rendered_location_text')]:
            assert r'\textbf{'+heading+'.}' in bodies[0]
            assert escape(row[key]) in bodies[0],row['id']
        for path in row['evidence_paths']:assert (ROOT/path).is_file(),path
        assert row['compiled_locations']=={a:locations[a] for a in row['manuscript_anchors']}
    ledger=read_json(OUT/'gpu/ledger.json');assert ledger['ceiling_seconds']==21600
    assert all(j['status']=='completed' and j['exit_code']==0 for j in ledger['jobs'])
    gpu_seconds=sum(j['charged_seconds'] for j in ledger['jobs']);assert gpu_seconds<=21600
    jobs=sorted(ledger['jobs'],key=lambda j:j['started_epoch'])
    assert all(a['finished_epoch']<=b['started_epoch'] for a,b in zip(jobs,jobs[1:]))
    for path in (OUT/'execution').glob('*.json'):
        receipt=read_json(path)
        # Worker receipts and supervised stderr independently retain CUDA evidence.
        assert receipt['n_gpu_layers'] == -1 and receipt['gpu_uuid'] == jobs[0]['gpu_uuid']
        stderr=(OUT/'gpu'/(path.stem+'.stderr.txt')).read_text()
        assert re.search(r'offloaded \d+/\d+ layers to GPU',stderr)
        assert 'NVIDIA RTX 5000 Ada Generation' in stderr
    builds=read_json(OUT/'validation/latex/latest_builds.json')
    assert set(builds)=={'scientific','correspondence','submission'}
    for phase,record in builds.items():
        for doc in record['records']:
            assert not doc['warnings'] and not doc['unresolved_references'],doc
    atomic_json(OUT/'validation/current_validation.json',{'status':'passed','baseline_commit':initial['head'],
        'protected_files_checked':len(initial['protected_git_blobs'])-len(ALLOWED_EXISTING_CHANGES),
        'exact_review_blocks':blocks,'written_answers':16,'unique_judge_requests':len(seen),'judge_attempts':attempts,
        'actual_judging_tokens':dict(actual_tokens),'gpu_job_wall_seconds':gpu_seconds,'one_gpu_process_at_a_time':True,
        'empirical_coverage':status['empirical_R1_4'],'published_current_study_coverage':False})
    print('Current-study preservation, scores, quotes, statuses and builds passed')


if __name__=='__main__':main()
