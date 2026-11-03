"""Input semantics must match saved weights even when tensor shapes match."""
from copy import deepcopy

KEYS = ('channel_policy', 'source_image_size', 'image_size', 'crop_mm',
        'percentile_low', 'percentile_high', 'slots', 'token_budget',
        'center_policy', 'trim_each_side', 'slot_weights', 'anatomy_features')


def input_contract(config):
    data = config['data']
    return dict(data={k: deepcopy(data.get(k)) for k in KEYS},
                token_metadata=bool(config['model'].get('token_metadata', False)))


def require_input_contract(config, checkpoint):
    if 'config' not in checkpoint:
        if config['data'].get('anatomy_features'):
            raise ValueError('Canonical inference requires checkpoint input metadata')
        return
    if input_contract(config) != input_contract(checkpoint['config']):
        raise ValueError('Inference input semantics differ from checkpoint configuration')
