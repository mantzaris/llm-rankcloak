"""Stage-aware validation with strict original review and retained-evidence invariants."""
import argparse,hashlib,re
from pathlib import Path
from collections import Counter
from rankcloak.revision_v4_stage2_common import ROOT,OUT,read_json,read_jsonl,file_hash,digest,atomic_json
from scripts.prepare_revision_v4_stage1 import escape


def verify_quotes(original,ledger,response):
    if ''.join(b['text'] for b in ledger['blocks'])!=original:raise ValueError('original review reconstruction changed')
    expected=['editor_letter','reviewer2','reviewer1_general']+[f'R1.{i}' for i in range(1,11)]
    if [b['id'] for b in ledger['blocks']]!=expected:raise ValueError('review blocks missing or reordered')
    for b in ledger['blocks']:
        pattern=r'% BEGIN QUOTE '+re.escape(b['id'])+r'\n\\begin\{reviewcomment\}\n(.*?)\n\\end\{reviewcomment\}\n% END QUOTE '+re.escape(b['id'])
        if re.findall(pattern,response,re.S)!=[escape(b['text'].rstrip())]:raise ValueError('verbatim review changed '+b['id'])
    return len(expected)


def verify_answers(response,ledger,allow_preview=False):
    if ledger['preview_execution_pending'] and not allow_preview:raise ValueError('preview answers still pending execution')
    if len(ledger['response_sets'])!=16:raise ValueError('sixteen response sets required')
    texts='\n'.join(p.read_text() for p in (ROOT/'paperV4/scientific_reports').rglob('*.tex'))
    anchors=set(re.findall(r'\\label\{([^}]+)\}',texts))
    for row in ledger['response_sets']:
        matches=re.findall(r'% BEGIN ANSWER '+re.escape(row['id'])+r'\n(.*?)% END ANSWER '+re.escape(row['id']),response,re.S)
        if len(matches)!=1:raise ValueError('answer missing '+row['id'])
        block=matches[0]
        for heading,key in [('Direct answer.','direct_answer'),('Evidence and change.','evidence_change'),('Locations.','location_text')]:
            if r'\textbf{'+heading+'}' not in block or escape(row[key]) not in block:raise ValueError('answer does not match evidence ledger '+row['id'])
        if row['status'] not in ['complete','partial','pending']:raise ValueError('unknown response status')
        if row['status']!='complete' and (not row['pending_work'] or 'INTERNAL PENDING WORK' not in block):raise ValueError('unresolved answer concealed')
        for name in row['evidence_paths']:
            if not (ROOT/name).is_file():
                if allow_preview and name.endswith('prospective_inventory.csv'):continue
                raise ValueError('missing answer evidence '+name)
        if not set(row['manuscript_anchors']).issubset(anchors):raise ValueError('nonexistent manuscript location '+row['id'])
    if 'Direct answer pending.' in response:raise ValueError('obsolete Stage 1 answer scaffold')
    return dict(Counter(row['status'] for row in ledger['response_sets']))


def verify_file_manifest(folder,field='files'):
    manifest=read_json(folder/('freeze.json' if (folder/'freeze.json').exists() else 'inventory_manifest.json'))
    for name,sha in manifest[field].items():
        if file_hash(folder/name)!=sha:raise ValueError('frozen input changed '+str(folder/name))


def verify_transport_reuse(folder):
    plan=read_json(folder/'reuse_validation_plan.json')
    expected=set(plan['request_ids']);rows=[]
    if len(expected)!=plan['requests']:raise ValueError('duplicate planned reuse identity')
    for model,count in plan['per_model'].items():
        result=read_json(folder/('reuse_validation_summary_'+model+'.json'))
        records=read_jsonl(folder/('reuse_validation_'+model+'.jsonl'))
        if result['planned_requests']!=count or result['completed_requests']!=count or len(records)!=count:
            raise ValueError('incomplete transport reuse validation')
        if result['source_sha256']!=plan['source_sha256'] or not result['all_agree']:
            raise ValueError('transport reuse validation contract or agreement changed')
        if any(not r['historical_rank_and_error_agreement'] for r in records):
            raise ValueError('transport reused endpoint differs')
        rows.extend(records)
    if len(rows)!=len(expected) or {r['request_id'] for r in rows}!=expected:
        raise ValueError('transport reuse request join mismatch')
    return len(rows)


def verify_gpu_accounting(ledger,unadmitted):
    jobs=ledger['jobs']
    if ledger['ceiling_seconds']!=28800 or sum(j['charged_seconds'] for j in jobs)>28800:
        raise ValueError('GPU ceiling exceeded')
    if any(j['charged_seconds']<0 for j in jobs):raise ValueError('negative GPU charge')
    active=[j for j in jobs if 'pid' in j]
    completed=sorted(active,key=lambda j:j['started_epoch'])
    for before,after in zip(completed,completed[1:]):
        if 'finished_epoch' not in before or before['finished_epoch']>after['started_epoch']:
            raise ValueError('overlapping supervised GPU jobs')
    for row in unadmitted['invocations']:
        matched=[j for j in jobs if j['name']==row['name']]
        if len(matched)!=1 or matched[0]['charged_seconds']<row['conservatively_charged_command_wall_seconds']:
            raise ValueError('unsuccessful invocation missing from conservative accounting')
    return sum(j['charged_seconds'] for j in jobs)


def main(preview=False):
    initial=read_json(OUT/'provenance/initial_state.json');changed=[]
    for name,sha in initial['protected_git_blobs'].items():
        p=ROOT/name
        if not p.exists():changed.append(name);continue
        raw=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual!=sha:changed.append(name)
    if changed:raise ValueError('historical or unrelated input changed '+str(changed))
    if file_hash(ROOT/'paperV4/RankCloak_V4_Stage2_Plan.md')!=initial['stage2_author_plan_sha256']:raise ValueError('author plan changed')
    request=ROOT/'paperV4/response/requests.txt'
    if file_hash(request)!=initial['requests_sha256']:raise ValueError('requests file changed')
    response=(ROOT/'paperV4/response/response_to_reviewers_v4.tex').read_text();quote_count=verify_quotes(request.read_text(),read_json(ROOT/'paperV4/response/review_comments.json'),response)
    answers=verify_answers(response,read_json(OUT/'response_evidence.json'),preview)
    for folder in ['initial','initial_unscored_roundtrip_preflight','amendment1','pilot_initial','pilot_amendment1','coherence_study']:verify_file_manifest(OUT/'plans'/folder)
    verify_file_manifest(OUT/'transport')
    # Source hashes bind original records separately from derived outputs.
    for folder in ['initial','amendment1']:
        r=read_json(OUT/'plans'/folder/'inventory_manifest.json')
        if r['instrumentation_source_sha256']!=file_hash(ROOT/'rankcloak/revision_v4_coherence.py'):raise ValueError('instrumentation source changed after freeze')
        if r['config_sha256']!=file_hash(ROOT/r['config_path']):raise ValueError('instrumentation configuration changed after freeze')
        for name,sha in r['source_hashes'].items():
            if file_hash(ROOT/name)!=sha:raise ValueError('inventory source changed')
    pilot=[read_json(OUT/'analysis'/p/'summary.json') for p in ['pilot_initial','pilot_amendment1']]
    if any(r['validated_metrics']!=['context_gain'] or r['all_primary_validated'] for r in pilot):raise ValueError('pilot decision changed')
    quant=read_json(OUT/'quantization/manifest.json')
    if (quant['pairs'],quant['positions'],quant['observed_rank_changes'])!=(1920,244440,69528):raise ValueError('quantization denominator mismatch')
    from rankcloak.revision_v4_stage2_scoring import scoring_contract
    contracts=read_json(OUT/'plans/coherence_study/scoring_contracts.json')
    if any(scoring_contract(k)!=contracts[k] for k in ['semantic','lm']):raise ValueError('scorer changed after freeze')
    filter_source=(ROOT/'results/revision_v4/manuscript_tables/filter_methods.tex').read_text().replace(r'\textbf{V4 working draft.} ','')
    if filter_source!=(ROOT/'paperV4/scientific_reports/v4_filter_methods.tex').read_text():raise ValueError('filter rule export changed')
    fulltext='\n'.join(p.read_text() for p in (ROOT/'paperV4/scientific_reports').glob('*.tex'))
    if 'computed filtered next-token Shannon entropy' in fulltext:raise ValueError('obsolete entropy filter claim')
    jobs=read_json(OUT/'gpu/ledger.json')['jobs'];charged=sum(r['charged_seconds'] for r in jobs)
    if charged>28800:raise ValueError('GPU ceiling exceeded')
    if not preview:
        if any(r['status']=='running' for r in jobs):raise ValueError('GPU work still running')
        verify_gpu_accounting(read_json(OUT/'gpu/ledger.json'),read_json(OUT/'gpu/unadmitted_invocations.json'))
        if verify_transport_reuse(OUT/'transport')!=53:raise ValueError('transport reuse count changed')
        summary=read_json(OUT/'analysis/coherence_study/summary.json');freeze=read_json(OUT/'plans/coherence_study/freeze.json')
        if summary['boundaries']!=freeze['selected_eligible_boundaries'] or summary['missing_planned_requests']:raise ValueError('incomplete frozen study')
        from rankcloak.revision_v4_stage2_analysis import join_scores
        scores=join_scores(OUT/'plans/coherence_study')
        if len(scores)!=summary['scoring_units']:raise ValueError('score denominator mismatch')
        claims=read_json(OUT/'analysis/coherence_study/manuscript_claims.json')
        if claims['source_summary_sha256']!=file_hash(OUT/'analysis/coherence_study/summary.json'):raise ValueError('stale manuscript numbers')
        for row in claims['primary_effects']:
            table=(ROOT/'paperV4/scientific_reports/supplementary_tables/v4_coherence.tex').read_text()
            if not all(f'{row[k]:.4f}' in table for k in ['effect','ci_low','ci_high']):raise ValueError('manuscript effect differs')
        model_table=(ROOT/'paperV4/scientific_reports/supplementary_tables/v4_coherence_models.tex').read_text()
        model_labels={'llama3_8b_instruct_q4_k_m':'Llama','qwen2_5_7b_instruct_q4_k_m':'Qwen','mistral_7b_instruct_v0_3_q4_k_m':'Mistral'}
        for model,label in model_labels.items():
            for contrast in ['actual minus ordinary','actual minus shuffled']:
                values=[next(r for r in summary['effects'] if r['model_id']==model and r['metric']==metric and r['contrast']==contrast) for metric in ['context_gain','conditional_logp']]
                expected=label+' & '+contrast+' & '+' & '.join(f"{r['effect']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]" for r in values)
                if expected not in model_table:raise ValueError('model-specific manuscript effect differs')
        secondary=next(r for r in summary['effects'] if r['model_id']=='all' and r['metric']=='conditional_logp' and r['contrast']=='actual minus ordinary')
        prose=(ROOT/'paperV4/scientific_reports/v4_stage2_results.tex').read_text()
        if not all(f'{secondary[k]:.4f}' in prose for k in ['effect','ci_low','ci_high']):raise ValueError('secondary manuscript effect differs')
        for name in ['main4','supplementary4','response_to_reviewers_v4','cover_letter_v4']:
            text=(OUT/'validation/latex'/(name+'_pass'+('4' if name in ['main4','supplementary4'] else '3')+'.txt')).read_text()
            if 'undefined references' in text or re.search(r'(Citation|Reference).*undefined',text):raise ValueError('unresolved document references')
        report=(ROOT/'revision_docs/REVISION_V4_STAGE2_REPORT.md').read_text()
        if 'INTERNAL EXECUTION DRAFT' in report or re.search(r'(?m)^[A-Z_]+_PENDING$',report):raise ValueError('Stage 2 report still has execution placeholders')
        resources=read_json(OUT/'gpu/resource_summary.json')
        if resources['ledger_sha256']!=file_hash(OUT/'gpu/ledger.json') or resources['gpu_job_wall_seconds']!=charged:raise ValueError('stale resource summary')
        build=read_json(OUT/'validation/latex/build_status.json')
        if len(build)!=4 or any(r['status']!='compiled' for r in build):raise ValueError('documents not built')
    result={'status':'preview passed' if preview else 'passed','protected_files_unchanged':len(initial['protected_git_blobs']),'requests_sha256':file_hash(request),'author_plan_unchanged':True,'exact_review_blocks':quote_count,'response_status_counts':answers,'gpu_job_wall_seconds':charged,'gpu_ceiling_seconds':28800,'stage1_outputs_unchanged':True,'frozen_plan_files_verified':True,'scoring_contracts_unchanged':True,'quantization_pairs':1920,'quantization_positions':244440,'transport_context_capacity_rechecks':None if preview else 53}
    atomic_json(OUT/'validation'/('preview_validation.json' if preview else 'stage2_validation.json'),result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');main(p.parse_args().preview)
