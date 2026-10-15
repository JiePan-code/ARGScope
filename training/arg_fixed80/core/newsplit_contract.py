"""Shared split and artifact provenance contract; no model code."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from fixed_split import FixedSplit, EXPECTED_FINGERPRINT

SCHEMA = 'argscope-fixed80-grouped-artifact-v1'
AUTHORITY = None
CHAIN = None

def configure(split_dir, chain):
    global AUTHORITY, CHAIN
    if chain not in ('full', 'wo_gate'):
        raise ValueError('Unknown upstream fusion chain')
    AUTHORITY, CHAIN = FixedSplit(split_dir), chain

def require_authority():
    if AUTHORITY is None:
        raise RuntimeError('Use the run_pipeline.py / BAT entrypoints; direct core scripts are disabled.')
    return AUTHORITY

def canonical_id(value):
    text = str(value).strip()
    return text.split()[0] if text else ''

def is_nonarg(value):
    compact = ''.join(ch for ch in str(value).strip().lower() if ch.isalnum())
    return compact in {'nonarg', 'nonresistance', 'negative', 'nonresistant'}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_ids(path):
    values = [canonical_id(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'Empty/duplicate authoritative IDs: {path}')
    return set(values)

def build_model_splits_from_new_ids(ids, subtypes, train_ids_set, test_ids_set, validation_fraction, seed):
    return require_authority().aligned_splits(ids, subtypes, train_ids_set, test_ids_set, validation_fraction, seed)

def type_mapping(path, subtypes):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        sample = stream.read(8192)
        stream.seek(0)
        delimiter = '\t' if '\t' in sample.splitlines()[0] else ','
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not {'Subtype', 'Type'}.issubset(reader.fieldnames or []):
            raise ValueError('Structure must contain Subtype and Type')
        mapping = {}
        for row in reader:
            label, typ = row['Subtype'].strip(), row['Type'].strip()
            if is_nonarg(label):
                continue
            if not label or not typ:
                raise ValueError('Empty Subtype/Type in structure')
            if label in mapping and mapping[label] != typ:
                raise ValueError(f'Conflicting Types for {label}')
            mapping[label] = typ
    wanted = {s for s in subtypes if not is_nonarg(s)}
    if wanted - mapping.keys():
        raise ValueError(f'Missing Type mapping: {sorted(wanted - mapping.keys())[:10]}')
    return {s: mapping[s] for s in sorted(wanted)}

def manifest(obj):
    return {key: obj[key] for key in ('ids', 'subtypes', 'external_splits', 'model_splits', 'exact_ids', 'subtype_to_type')}

def fingerprint(obj):
    return hashlib.sha256(json.dumps(manifest(obj), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def stamp_fusion(obj, structure, train_path, test_path, output_path):
    require_authority().validate_structure(structure)
    obj['schema'] = SCHEMA
    obj['authoritative_split_fingerprint'] = EXPECTED_FINGERPRINT
    obj['fusion_chain'] = CHAIN
    obj['external_splits'] = list(obj['splits'])
    obj['exact_ids'] = {s: [x for x, split in zip(obj['ids'], obj['model_splits']) if split == s] for s in ('train', 'validation', 'test')}
    obj['subtype_to_type'] = type_mapping(structure, obj['subtypes'])
    obj['split_fingerprint'] = fingerprint(obj)
    obj['authoritative_id_sha256'] = {'train': sha256(train_path), 'test': sha256(test_path)}
    validate_fusion(obj, train_path, test_path)
    folder = Path(output_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    for split, ids in obj['exact_ids'].items():
        (folder / f'{split}_ids.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
    (folder / 'split_manifest.json').write_text(json.dumps(manifest(obj), indent=2), encoding='utf-8')

def validate_fusion(obj, train_path=None, test_path=None):
    authority = require_authority()
    if (obj.get('authoritative_split_fingerprint') != EXPECTED_FINGERPRINT or
        obj.get('fusion_chain') != CHAIN):
        raise ValueError('Wrong split or upstream chain; old artifacts are forbidden')
    if obj.get('schema') != SCHEMA or obj.get('split_fingerprint') != fingerprint(obj):
        raise ValueError('Old or inconsistent fusion artifact: regenerate stage 01')
    ids, subtypes = obj['ids'], obj['subtypes']
    train = {x for x, s in zip(ids, obj['external_splits']) if s == 'train'}
    test = {x for x, s in zip(ids, obj['external_splits']) if s == 'test'}
    if train_path is not None and (read_ids(train_path) != train or read_ids(test_path) != test):
        raise ValueError('Fusion differs from authoritative train/test ID files')
    model, external = build_model_splits_from_new_ids(ids, subtypes, train, test, .20, 20260822)
    if obj['model_splits'] != model or obj['external_splits'] != external or obj['splits'] != external:
        raise ValueError('Saved split differs from deterministic Student rule')
    for split in ('train', 'validation', 'test'):
        if obj['exact_ids'][split] != [x for x, s in zip(ids, model) if s == split]:
            raise ValueError('Exact ID manifest mismatch')
    train_classes = {s for s, split in zip(subtypes, model) if split == 'train'}
    if set(subtypes) - train_classes:
        raise ValueError('Subtype absent from model-train')
    if obj['subtype_to_type'] != authority.type_mapping:
        raise ValueError('Type mapping differs from fixed manifest')
    report(obj)
    return model, external

def report(obj):
    print(f"Split counts: {dict(Counter(obj['model_splits']))} | ARG subtype count: {len(obj['subtype_to_type'])} | Type count: {len(set(obj['subtype_to_type'].values()))}", flush=True)

def use_saved_splits(obj, ids, subtypes, train_ids_set, test_ids_set, validation_fraction, seed):
    model, external = validate_fusion(obj)
    if ids != obj['ids'] or subtypes != obj['subtypes']:
        raise ValueError('Student/fusion ID order or labels differ')
    expected, ext = build_model_splits_from_new_ids(ids, subtypes, train_ids_set, test_ids_set, validation_fraction, seed)
    if model != expected or external != ext:
        raise ValueError('Student authoritative IDs differ from fusion')
    return list(model), list(external)

def stamp_derived(output, fused, fusion_path, kind):
    validate_fusion(fused)
    output['schema'] = SCHEMA
    output['authoritative_split_fingerprint'] = EXPECTED_FINGERPRINT
    output['fusion_chain'] = CHAIN
    output['artifact_kind'] = kind
    output['split_manifest'] = manifest(fused)
    output['split_fingerprint'] = fused['split_fingerprint']
    output['source_fusion_sha256'] = sha256(fusion_path)

def validate_derived(obj, fused, fusion_path, kind):
    validate_fusion(fused)
    if (obj.get('schema') != SCHEMA or obj.get('artifact_kind') != kind or
        obj.get('authoritative_split_fingerprint') != EXPECTED_FINGERPRINT or obj.get('fusion_chain') != CHAIN or
        obj.get('split_fingerprint') != fused['split_fingerprint'] or
        obj.get('split_manifest') != manifest(fused) or
        obj.get('source_fusion_sha256') != sha256(fusion_path)):
        raise ValueError(f'Old/mismatched {kind}: regenerate upstream stages')
    if set(obj['arg_subtype_names']) != set(fused['subtype_to_type']):
        raise ValueError('Prototype subtype set differs from fusion')
