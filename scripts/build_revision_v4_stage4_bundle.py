"""Build a self-contained upload bundle with Stage 4 logs and explicit publication status.

Pending mode preserves every current manuscript byte. Final mode requires a verified
public archive and already finalized document sources. This tool never edits them.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from scripts.prepare_revision_v4_stage3_handoff import flatten
from scripts.prepare_revision_v4_stage4 import ROOT, OUT, SOURCE, sha, write_json

DOCS = [('scientific_reports', 'main4', True), ('scientific_reports', 'supplementary4', True),
        ('response', 'response_to_reviewers_v4', False), ('cover_letter', 'cover_letter_v4', False)]
DEPENDENCIES = ['wlscirep.cls', 'naturemag-doi.bst', 'jabbrv.sty', 'jabbrv-ltwa-all.ldf', 'jabbrv-ltwa-en.ldf']


def manuscript_source(folder):
    source = flatten(folder / 'main4.tex')
    source = source.replace(r'\bibliography{references}', (folder / 'main4.bbl').read_text())
    figures = re.findall(r'\\begin\{figure\}\[H\].*?\\end\{figure\}', source, re.S)
    tables = re.findall(r'\\begin\{table\}\[H\].*?\\end\{table\}', source, re.S)
    legends = []
    for block in figures:
        source = source.replace(block, '')
        legends.append('\\begin{figure}[H]\n' + block[block.index('\\caption'):block.rindex('\\end{figure}')] + '\\end{figure}\n')
    for block in tables:
        source = source.replace(block, '')
    source = source.replace(r'\end{document}', '\\section*{Figure legends}\n' + ''.join(legends) + '\n\\section*{Tables}\n' + '\n'.join(tables) + '\n\\end{document}')
    return source


def pdf_text(path):
    return subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)


def pages(path):
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    return int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])


def dependency_check(fls, allowed_root):
    # All project inputs must be local. System TeX distribution inputs are expected.
    system = [Path(p).resolve() for p in ['/usr/share/texlive', '/usr/share/texmf', '/var/lib/texmf', '/etc/texmf']]
    checked = set()
    for line in fls.read_text().splitlines():
        if not line.startswith('INPUT '):
            continue
        raw = Path(line[6:])
        p = (raw if raw.is_absolute() else fls.parent / raw).resolve()
        if p in checked:
            continue
        checked.add(p)
        if not p.is_relative_to(allowed_root.resolve()) and not any(p.is_relative_to(d) for d in system):
            raise ValueError('Build input outside bundle or TeX distribution: ' + str(p))
    return len(checked)


def compile_document(directory, name, bibliography, logs, allowed_root):
    start = time.monotonic()
    cmd = ['pdflatex', '-recorder', '-interaction=nonstopmode', '-halt-on-error', name + '.tex']
    commands = [cmd] + ([['bibtex', name]] if bibliography else []) + [cmd, cmd]
    env = os.environ.copy()
    for key in ['TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS']:
        env.pop(key, None)
    env['TEXMFHOME'] = str(allowed_root / 'empty_texmf')
    for i, command in enumerate(commands, 1):
        run = subprocess.run(command, cwd=directory, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=120)
        (logs / f'{name}_pass{i}.txt').write_text(run.stdout)
        if run.returncode:
            raise ValueError('LaTeX build failed: ' + name)
    warnings = [s for s in run.stdout.splitlines() if any(x in s for x in ('Warning:', 'Overfull', 'Underfull'))]
    if warnings:
        raise ValueError('Final build warnings: ' + str(warnings))
    count = dependency_check(directory / (name + '.fls'), allowed_root)
    return {'name': name, 'pages': pages(directory / (name + '.pdf')), 'warnings': warnings,
            'input_paths_checked': count, 'elapsed_seconds': time.monotonic() - start,
            'pdf_sha256': sha(directory / (name + '.pdf'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pending', 'final'], default='pending')
    parser.add_argument('--directory', type=Path, required=True, help='Fresh parent directory under ignored release_artifacts')
    args = parser.parse_args()
    folder = ROOT / 'paperV4/scientific_reports'
    if args.directory.exists():
        raise ValueError('Use a fresh directory to preserve previous build evidence')
    if not args.directory.resolve().is_relative_to((ROOT / 'release_artifacts').resolve()):
        raise ValueError('Place large/local bundle artifacts under release_artifacts')
    if args.mode == 'final':
        receipt = json.loads((OUT / 'publication/PUBLIC_VERIFICATION_RECEIPT.json').read_text())
        if receipt['status'] != 'passed' or receipt['source_commit'] != SOURCE:
            raise ValueError('Final bundle requires verified publication of the approved snapshot')
        for directory, name, _ in DOCS:
            if 'V4 AUTHOR-REVIEW CANDIDATE' in (ROOT / 'paperV4' / directory / (name + '.tex')).read_text():
                raise ValueError('Finalize approved document labels before building a final bundle')
        if receipt['version_doi'] not in (folder / 'main4.tex').read_text():
            raise ValueError('Final manuscript lacks the verified DOI')
    submission = (folder / 'main4_submission.tex').read_text()
    if submission.split('\n', 1)[1] != manuscript_source(folder):
        raise ValueError('Single-file manuscript is stale. Regenerate from finalized sources before bundling.')
    if re.search(r'\\(?:input|includegraphics|bibliography)\{', submission):
        raise ValueError('Upload manuscript has unresolved external content')
    title = re.search(r'\\title\{([^}]+)\}', submission)[1]
    if title != re.search(r'\\title\{([^}]+)\}', (folder / 'supplementary4.tex').read_text())[1]:
        raise ValueError('Title differs between manuscript and supplement')
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', flatten(folder / 'main4.tex'))
    if len(figures) != 5 or submission.count(r'\begin{table}') != 3:
        raise ValueError('Display item inventory changed')
    bundle = args.directory / ('submission_pending_publication' if args.mode == 'pending' else 'submission_final')
    bundle.mkdir(parents=True)
    logs = OUT / 'validation' / ('build_' + args.directory.name)
    logs.mkdir(parents=True, exist_ok=False)
    mappings = []
    def add(source, dest, role):
        target = bundle / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        mappings.append({'filename': dest, 'portal_role': role, 'source': str(source.relative_to(ROOT)),
                         'size_bytes': target.stat().st_size, 'sha256': sha(target)})
    add(folder / 'main4_submission.tex', 'main4_submission.tex', 'Editable revised manuscript')
    for name in DEPENDENCIES:
        add(folder / name, name, 'LaTeX dependency')
    add(folder / 'main4.bbl', 'main4.bbl', 'Compiled bibliography companion, already embedded in main source')
    for i, name in enumerate(figures, 1):
        if pages(folder / name) != 1:
            raise ValueError('Main figure is not single-page: ' + name)
        add(folder / name, name, f'Main Figure {i}, separate single-page upload')
    for directory, name, _ in DOCS:
        role = {'main4': 'Author reference only, not the editable manuscript slot',
                'supplementary4': 'Supplementary Information', 'response_to_reviewers_v4': 'Response to Reviewers, single file',
                'cover_letter_v4': 'Cover letter'}[name]
        add(ROOT / 'paperV4' / directory / (name + '.pdf'), name + '.pdf', role)
    status = 'PENDING PUBLICATION. Do not upload this directory yet.' if args.mode == 'pending' else 'Prepared for author portal upload. No journal submission performed.'
    readme = ['# Journal upload map', '', status, '',
              'The research ZIP belongs on Zenodo. It is not the journal manuscript file.',
              'The manuscript is editable single-column LaTeX with compiled references, end legends and editable tables.',
              'Five figures and three tables are counted separately from three algorithm floats. Confirm the portal classification if it requests a combined display count.',
              '', '| Filename | Portal role |', '| --- | --- |']
    readme += ['| ' + r['filename'] + ' | ' + r['portal_role'] + ' |' for r in mappings]
    (bundle / 'UPLOAD_MAP.md').write_text('\n'.join(readme) + '\n')
    mapping = {'status': args.mode, 'public_v4_verified': args.mode == 'final', 'journal_submission_performed': False,
               'bundle_directory': str(bundle.resolve()), 'source_snapshot_commit': SOURCE, 'files': mappings}
    write_json(bundle / 'UPLOAD_MAP.json', mapping)
    write_json(OUT / 'upload_map.json', mapping)
    (OUT / 'UPLOAD_MAP.md').write_text('\n'.join(readme) + '\n')
    clean = args.directory / 'clean_upload_build'
    shutil.copytree(bundle, clean)
    builds = [compile_document(clean, 'main4_submission', False, logs, clean)]
    # Separately rebuild all four current documents in a clean copy, never in paperV4.
    work = args.directory / 'clean_document_build'
    tracked = subprocess.check_output(['git', 'ls-files', 'paperV4'], cwd=ROOT, text=True).splitlines()
    for name in tracked:
        source = ROOT / name
        if source.suffix not in ['.tex', '.cls', '.bst', '.sty', '.ldf', '.bib', '.bbl', '.pdf', '.png']:
            continue
        target = work / Path(name).relative_to('paperV4')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for directory, name, bib in DOCS:
        result = compile_document(work / directory, name, bib, logs, work)
        original = ROOT / 'paperV4' / directory / (name + '.pdf')
        rebuilt = work / directory / (name + '.pdf')
        result['source_pdf_sha256'] = sha(original)
        result['text_layout_equal_to_current_pdf'] = pdf_text(original) == pdf_text(rebuilt)
        if not result['text_layout_equal_to_current_pdf']:
            raise ValueError('Current PDF is stale relative to its source: ' + name)
        builds.append(result)
    write_json(logs / 'build_status.json', {'status': 'passed', 'mode': args.mode, 'builds': builds,
               'science_rerun': False, 'gpu_jobs': 0, 'pdf_hash_caveat': 'Fresh TeX timestamps may differ. All four text layouts exactly match the current PDFs.'})
    print(json.dumps({'status': 'passed', 'mode': args.mode, 'bundle': str(bundle), 'build_pages': {r['name']: r['pages'] for r in builds}}, indent=2))


if __name__ == '__main__':
    main()
