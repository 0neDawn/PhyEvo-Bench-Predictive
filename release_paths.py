"""Portable filesystem locations for the release copy; no model operations."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent


def project_path(relative=""):
    return str(ROOT / relative)


def verified_source_revision(relative):
    """Verify packaged source bytes without depending on a checkout's .git folder."""
    components = json.loads((ROOT / 'configs/third_party.json').read_text())
    component = next(entry for entry in components if entry['directory'] == relative)
    revision = component['revision']
    if not revision:
        raise RuntimeError('No pinned source revision for ' + relative)
    manifest = json.loads((ROOT / 'audit/source_manifest.json').read_text())
    files = [entry for entry in manifest if entry['distribution'] == 'included'
             and entry['path'].startswith(relative.rstrip('/') + '/')]
    if not files:
        raise RuntimeError('No source hashes recorded for ' + relative)
    for entry in files:
        observed = hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest()
        if observed != entry['release_sha256']:
            raise RuntimeError('Packaged source checksum mismatch: ' + entry['path'])
    return revision


def runtime_python(kind="other"):
    key = 'PHYEVO_DDLP_PYTHON' if kind == 'ddlp' else 'PHYEVO_PYTHON'
    if os.environ.get(key):
        return os.environ[key]
    local = ROOT / 'configs/runtime_interpreters.local.json'
    if local.is_file():
        value = json.loads(local.read_text()).get(kind)
        if value:
            return value
    return sys.executable
