"""Explicit RSNA public NPZ protocol, preserving the producer's actual basis.

Transport/schema inspection is deliberately separate from training admission.
This candidate never substitutes a transport receipt for a semantic release.
"""
from pathlib import Path
import json, math, hashlib, os

STATE = 'canonical_L_v8_3_with_author_v4_basis_adapter'
COLS = ['series_idx', 'slot', 'plane', 'view_id', 'rel_pos', 'log_spacing',
        'extent_130', 'fluid', 'fatsat', 'slice_order', 'n_slices', 'real_series_idx']
VIEWS = {'Sagittal': {0, 1, 4, 5}, 'Coronal': {0, 2, 3}, 'Axial': {0, 1, 4, 5}}

def need(value, code):
    if not value: raise ValueError('RSNA_PUBLIC_NPZ:' + code)

def documents(bundle):
    files = bundle['files']
    def one(suffix):
        found = [(k, v) for k, v in files.items() if k.endswith(suffix)]
        need(len(found) == 1, 'EXACT_DOCUMENT:' + suffix)
        return found[0]
    return files, one('/PART_MANIFEST.json'), one('/CONFIG_BINDING.json'), one('/ANCHORS.json'), one('/NATIVE_PROVENANCE.json')

def inspect(bundle):
    files = bundle['files']; receiver = files['.receiver_verified.json']['document']
    target = receiver['target']
    need(target['instance_id'] == 56967269 and target['machine_id'] == 1569 and target['hostname'] == '56dc3869206e', 'TARGET')
    need(target['boot_id'] == bundle['target']['boot_id'] == 'd7c9a413-650b-4f6e-b872-2fa572010446', 'BOOT')
    need(receiver['pass'] is True and receiver['checked_fullSHA_count'] == receiver['file_count'], 'RECEIVER_FULLSHA_STATUS')
    manifest_record = files[receiver['manifest_relative_path']]
    need(receiver['manifest_sha256'] == manifest_record['sha256'], 'BATCH_MANIFEST_SHA')
    manifest = manifest_record['document']
    need(receiver['source_manifest'] == manifest, 'BATCH_CONTENT')
    need(manifest['file_count'] == len(manifest['files']) == receiver['file_count'], 'BATCH_FILE_COUNT')
    table = {v['relative_path']: v for v in manifest['files']}
    verified = receiver['verified_files']
    need(len(table) == len(manifest['files']) and set(table) == set(verified), 'EXACT_BATCH_MEMBERS')
    _, (partpath, partrec), (configpath, configrec), (anchorpath, anchorrec), (nativepath, nativerec) = documents(bundle)
    part, config, anchors, native = [v['document'] for v in [partrec, configrec, anchorrec, nativerec]]
    unit = str(Path(partpath).parent)
    sidepath = unit + '/' + part['renderer_sidecar_relative_path']; side = files[sidepath]['document']
    selected = [partpath] + [unit + '/' + v['relative_path'] for v in part['files']]
    need(len(selected) == 6 and part['file_count'] == len(part['files']) == 5 and len(set(selected)) == 6, 'UNIT_MEMBERS')
    for path in selected:
        need(path in table and path in verified, 'UNIT_NOT_IN_BATCH')
        source, dest = table[path], verified[path]
        need(source['sha256'] == dest['sha256'] and source['bytes'] == dest['bytes'], 'DESTINATION_SHA')
        if path in files:
            actual = files[path]
            need(actual['sha256'] == dest['sha256'], 'METADATA_SHA')
            need(actual['stat'] == [dest['stat']['size'], dest['stat']['mtime_ns'], dest['stat']['ctime_ns']] and actual['inode'] == dest['stat']['inode'], 'METADATA_STAT')
    for row in part['files']:
        need(row['sha256'] == table[unit + '/' + row['relative_path']]['sha256'] and row['bytes'] == table[unit + '/' + row['relative_path']]['bytes'], 'PART_MEMBER_SHA')
    need(part['status'] == 'UNIT_SEALED_FULL_LOCAL_SHA_VERIFIED' and side['completed'] is True and side['status'] == 'STUDY_COMMITTED_VERIFIED', 'COMMITTED_STATUS')
    need(side['schema'] == 'rsna-public-v4-compressed-study-v1', 'SCHEMA')
    for name in ['cache_identity', 'config_sha256', 'code_identity_sha256', 'source_identity_sha256', 'weights_sha256', 'run_id']:
        need(part[name] == config[name] == manifest[name], 'IDENTITY:' + name)
    need(part['SampleID'] == config['SampleID'] == side['SampleID'] == side['sample'], 'SAMPLE_ID')
    need(part['parent_study_uid'] == config['parent_study_uid'] == side['parent_study_uid'] == native['parent_study_uid'], 'PARENT_ID')
    need(part['report_group'] == config['report_group'] and bool(part['report_group']), 'REPORT_GROUP')
    need(part['partition'] == config['partition'] == side['partition'] and part['partition'] in ['fit', 'Gold'], 'PARTITION')
    orientation = side['orientation']
    need(part['orientation_state'] == orientation['state'] == side['config']['cache_identity'] == STATE, 'EXPLICIT_BASIS_STATE')
    need(native['orientation'] == orientation, 'NATIVE_ORIENTATION_BINDING')
    need(orientation['laterality_resolver_calls'] == 1 and orientation['legacy_side_classifier_used'] is False and orientation['author_native_basis'] is True, 'RESOLVER_ONCE')
    composition = orientation['synthetic_composition_proof']
    need(composition['status'] == 'PASS' and composition['identity'] == STATE and composition['exact_frozen_apply_tokens_parity'] is True and composition['depth_order_proof'] is True, 'COMPOSITION_DECLARATION')
    need(side['meta_cols'] == COLS, 'META_COLUMNS')
    px, meta = side['payload']['arrays']['px'], side['payload']['arrays']['meta']; rows = side['n_rows']
    need(px['shape'] == [rows, 384, 384] and px['dtype'] == 'uint8' and meta['shape'] == [rows, 12] and meta['dtype'] == 'float32', 'ARRAY_DECLARATION')
    need(rows == part['pixel_rows'] and side['payload']['sha256'] == part['payload_sha256'] and px['sha256'] == part['pixel_array_sha256'] and meta['sha256'] == part['meta_array_sha256'], 'PAYLOAD_BINDING')
    need(anchors['audit']['component_sha256s']['anatomy.pt'] == part['weights_sha256'], 'ANATOMY_WEIGHTS')
    spans = []; sops = set()
    for block in side['blocks']:
        n = block['n_slices']; plane = block['plane']; views = block['views']
        need(n > 0 and plane in VIEWS and {v for v, start in views} == VIEWS[plane] and len(views) == len(VIEWS[plane]), 'VIEWS')
        need(block['render_basis'] == 'author_v4' and len(block['source_groups']) == n, 'BLOCK_BASIS_OR_GROUPS')
        audit = block['basis_audit']
        need(audit['canonicalization_count'] == 1 and audit['geometry_composition_verified'] is True and audit['one_pixel_index_mapping'] is True and audit['no_legacy_native_side_flip'] is True, 'SERIES_COMPOSITION')
        thin = block['thin_stack']
        from rsna_slabs import validate as validate_public_slabs
        original=[v for v in native['series'] if v['SeriesInstanceUID']==block['series']]
        need(len(original)==1,'EXACT_NATIVE_SERIES')
        slab=validate_public_slabs(block,original[0])
        sops.update(slab['source_SOPs'])
        need(thin['target_slab_mm'] == 3.5 and thin['threshold_mm'] == 1.75, 'SLAB_POLICY')
        for group in block['source_groups']:
            weights = group['uniform_weights']; source = group['source_frames']
            need(len(weights) == len(source) == len(group['native_indices']) and len(weights) > 0 and all(math.isclose(w, 1 / len(weights), abs_tol=1e-9) for w in weights), 'SLAB_WEIGHTS')
            for frame in source:
                need(frame['crc_verified'] is True and frame['decode_valid'] is True, 'SOURCE_FRAME_VALIDITY')
                sops.add(frame['SOPInstanceUID'])
        for view, start in views:
            need(isinstance(start, int) and 0 <= start < start + n <= rows, 'ROW_RANGE')
            spans.append((start, start + n))
        need({v['view_id'] for v in block['render_views']} == VIEWS[plane], 'RENDER_VIEWS')
        for view in block['render_views']:
            need(view['render_basis'] == 'author_v4' and (view['crop_mm'] is None if view['view_id'] == 0 else view['crop_mm'] == 100.0), 'FULLFIELD_AND_100MM')
    spans.sort()
    need(spans[0][0] == 0 and spans[-1][1] == rows and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), 'EXACT_ROW_COVERAGE')
    from rsna_aliases import validate as validate_native_aliases
    sops.update(validate_native_aliases(side,native))
    coverage = part['actual_SOP_coverage']
    policy=json.loads((Path(__file__).parent/'RSNA_METADATA_POLICY.frozen.json').read_text())
    excluded=[q for q in policy['source_quarantine'] if q['sample']==part['SampleID']]
    from rsna_routes import validate as validate_frozen_routes
    parent_sop_count=validate_frozen_routes(part,side,native,bundle.get('routing_context'),sops)
    need(coverage['all_stored_layers_have_source_SOP_binding'] is True,'SELECTED_STUDY_SOURCE_ROUTES')
    need(coverage['excluded_frame_count']==len(excluded) and len(sops)==coverage['native_retained_SOP_count'] and coverage['source_member_count']==parent_sop_count+len(excluded),'SELECTED_STUDY_SOP_COVERAGE')
    original_members=[name for series in native['series'] for name in series['source_order_receipt']['files']]
    for item in excluded:
        need(item['pixels_never_decoded'] is True and item['original_source_index']==0 and item['SOPInstanceUID'] not in sops,'FROZEN_QUARANTINE_NOT_USED')
        matches=[member for member in original_members if member.endswith('/'+item['SOPInstanceUID']+'.dcm')]
        need(len(matches)==1 and '/'+part['parent_study_uid']+'/' in matches[0] and len(item['source_member_sha256'])==64,'FROZEN_QUARANTINE_SOURCE_BINDING')
    payloadpath = unit + '/' + part['payload_relative_path']
    need(table[payloadpath]['sha256'] == side['payload']['sha256'], 'BATCH_PAYLOAD_SHA')
    return {'status': 'PASS_SELECTED_UNIT_TRANSPORT_AND_DECLARED_PROTOCOL_ONLY', 'sample_id': part['SampleID'], 'parent_id': part['parent_study_uid'], 'report_group': part['report_group'], 'partition': part['partition'], 'orientation_state': STATE, 'resolver_to_apply_in_reader': False, 'rows': rows, 'series': len(side['blocks']), 'views': len(spans), 'source_SOPs': len(sops), 'payload_path': payloadpath, 'payload_destination_binding': verified[payloadpath], 'native_adapter_sha256': native['native_adapter_sha256'], 'source_code_identity_sha256': part['code_identity_sha256'], 'actual_payload_decoded': False, 'semantic_acceptance': False, 'fullcache_coverage': False, 'training_release': False, 'missing_before_release': ['final corpus coverage including test3/Gold and aliases', 'exact custom basis adapter code binding and actual geometry/rerender QA', 'full production reader/MainDataset and threecorpus CPU/GPU admission']}

def require_training_release(assessment,acceptance=None):
    # The original transport-only status does not itself become a semantic PASS.
    # Verify the frozen output-based dossier and this exact accepted unit.
    from actual_acceptance import require_unit_release
    return require_unit_release(assessment)
