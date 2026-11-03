"""One shared train/evaluation contract for optional token metadata."""


def model_inputs(batch, device):
    names = ('tokens', 'token_mask', 'slot_ids')
    if 'token_features' in batch:
        names += ('token_features',)
    return tuple(batch[k].to(device, non_blocking=True) for k in names)
