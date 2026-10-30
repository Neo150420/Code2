"""Validate a declared reconstruction without changing the trained algorithm."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contract(root, contract, previous=None):
    if contract['classification'] != 'corrected_author_asset_reconstruction':
        raise ValueError('Unsupported reproduction classification')
    for group in ('critical_source_sha256', 'asset_sha256', 'dependency_lock_sha256', 'execution_source_sha256'):
        for name, digest in contract.get(group, {}).items():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Contract path must stay within repository: {name}')
            if file_sha256(Path(root)/path) != digest:
                raise ValueError(f'Frozen reproduction input changed: {name}')
    if previous is not None:
        if previous['protocol'] != contract['protocol']:
            raise ValueError('Existing training protocol differs from contract')
        if previous['code_sha256'] != contract['critical_source_sha256']:
            raise ValueError('Existing training source differs from contract')
        for key, value in contract['fixed_cli'].items():
            if previous[key] != value:
                raise ValueError(f'Existing training setting differs: {key}')


def bind_contract(root, run, filename):
    """Keep an immutable, hashed copy alongside the experiment."""
    source = Path(filename).resolve()
    raw = source.read_bytes()
    contract = json.loads(raw)
    previous_file = Path(run)/'training/config.json'
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else None
    validate_contract(root, contract, previous)
    target = Path(run)/'reproduction-contract.json'
    if target.exists() and target.read_bytes() != raw:
        raise ValueError('Experiment already has a different frozen contract')
    if not target.exists():
        target.write_bytes(raw)
    return contract, hashlib.sha256(raw).hexdigest()
