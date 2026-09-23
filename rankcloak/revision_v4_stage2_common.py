"""Content identities and durable Stage 2 artifacts. Historical inputs are read only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from rankcloak.revision_artifacts import canonical_json_bytes, canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/stage2'
GPU = 'GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf'


def digest(value):
    return canonical_json_sha256(value)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('immutable artifact changed: ' + str(path))
        return
    atomic_json(path, value)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(canonical_json_bytes(row).decode() + '\n')


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(canonical_json_bytes(row).decode() + '\n'); f.flush(); os.fsync(f.fileno())


def hash_order(seed, identity):
    return hashlib.sha256(f'{seed}|{identity}'.encode()).hexdigest()


def load_cache(path, expected_contract):
    rows = read_jsonl(path) if Path(path).exists() else []
    cache = {}
    for row in rows:
        if row['contract_sha256'] != expected_contract:
            raise ValueError('checkpoint contract changed')
        identity = row['request_id']
        if identity in cache:
            raise ValueError('duplicate checkpoint identity')
        if digest(row['request']) != identity:
            raise ValueError('checkpoint request identity changed')
        cache[identity] = row
    return cache
