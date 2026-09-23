"""Canonical decoder compatibility checking. A fingerprint is not authentication."""
from __future__ import annotations
import hmac
from collections.abc import Mapping
from rankcloak.revision_artifacts import canonical_json_bytes, canonical_json_sha256

REQUIRED_FIELDS = frozenset({'model','tokenizer','quantization','backend','prompt_rendering','bos_eos_policy',
                            'ordering','filter','codec_framing','gate','boundary_reset'})
SCHEMA='rankcloak-decoder-configuration-v1'


def configuration_fingerprint(configuration):
    if not isinstance(configuration,Mapping) or set(configuration)!=REQUIRED_FIELDS:
        raise ValueError('configuration must contain the complete decoder contract')
    if any(not isinstance(configuration[k],Mapping) or not configuration[k] for k in REQUIRED_FIELDS):
        raise ValueError('each contract field must be a nonempty explicit object')
    # The existing canonical helper rejects nonfinite values and serializes keys stably.
    import json
    normalized=json.loads(canonical_json_bytes(dict(configuration)))
    body={'schema_version':SCHEMA,'configuration':normalized}
    return {**body,'configuration_sha256':canonical_json_sha256(body)}


def require_compatible(advertised,local_configuration):
    if advertised.get('schema_version')!=SCHEMA:raise ValueError('unsupported configuration fingerprint schema')
    expected=configuration_fingerprint(advertised['configuration'])
    if advertised!=expected:raise ValueError('corrupt advertised configuration fingerprint')
    local=configuration_fingerprint(local_configuration)
    if not hmac.compare_digest(expected['configuration_sha256'],local['configuration_sha256']):
        changed=sorted(k for k in REQUIRED_FIELDS if expected['configuration'][k]!=local['configuration'][k])
        raise ValueError('decoder configuration mismatch: '+', '.join(changed))
    return True
