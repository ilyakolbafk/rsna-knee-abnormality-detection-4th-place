"""Fresh dispatch and fresh-boot resume have different admission bindings."""
from common import TARGET,read,sha
from recovery_state import RecoveryStore


def require_run_intent(root,arm,package,boot,*,resume=False,binding=None):
    intent=read(root/'runtime/DISPATCH_INTENT.json')
    if intent['package_sha256']!=package or intent['target']!=TARGET or intent['arms']!=['core6','vm']:raise ValueError('EXCLUSIVE_PAIR_INTENT_REQUIRED')
    current=sha(root/'runtime/ADMISSION.json')
    if not resume:
        if intent['admission_sha256']!=current:
            auth=read(root/'runtime'/f'STARTUP_AUTHORIZATION_{arm}.json')
            if auth.get('status')!='PASS_UNSTARTED_ARM_CURRENT_BOOT_RECONCILIATION' or auth.get('arm')!=arm or auth.get('package_sha256')!=package or auth.get('original_intent_sha256')!=sha(root/'runtime/DISPATCH_INTENT.json') or auth.get('admission_sha256')!=current or auth.get('boot_id')!=boot or auth.get('smoke_sha256')!=sha(root/'runtime'/f'SMOKE_{arm}.json'):raise ValueError('FRESH_INTENT_ADMISSION_BINDING')
            if (root/'outputs'/arm).exists() or (root/'runtime'/f'STARTED_{arm}.json').exists():raise ValueError('UNSTARTED_RECONCILIATION_AFTER_START')
        return
    auth=read(root/'runtime'/f'RESUME_AUTHORIZATION_{arm}.json')
    if auth['status']!='PASS_FULLSHA_RESUME_AUTHORIZATION' or auth['arm']!=arm or auth['package_sha256']!=package or auth['original_intent_sha256']!=sha(root/'runtime/DISPATCH_INTENT.json') or auth['new_admission_sha256']!=current or auth['new_smoke_sha256']!=sha(root/'runtime'/f'SMOKE_{arm}.json') or auth['boot_id']!=boot:raise ValueError('FRESH_RESUME_AUTHORIZATION_REQUIRED')
    if auth['recovery_manifest'] not in ['slot0.manifest.json','slot1.manifest.json']:raise ValueError('INVALID_RECOVERY_COMMIT_PATH')
    path=root/'outputs'/arm/'recovery'/auth['recovery_manifest']
    if sha(path)!=auth['recovery_manifest_sha256']:raise ValueError('RESUME_SLOT_COMMIT_CHANGED')
    if binding is None:raise ValueError('RECOVERY_BINDING_REQUIRED')
    state,entry=RecoveryStore(root/'outputs'/arm/'recovery',binding).load()
    if f"slot{entry['slot']}.manifest.json"!=auth['recovery_manifest'] or entry['sha256']!=auth['slot_sha256']:raise ValueError('AUTHORIZED_RECOVERY_SLOT_CHANGED')
    del state
