"""Public full96 FM auxiliary forward and merged gradient, private RNG stream."""
import math
import torch
from torch.nn import functional as F
from fm_math import fm_photometric,fm_fork_seed

def backward(model, batch, coefficient=.3, chunk=96, *, seed_base, rng_key, epoch, step, device):
    """Public dense BCE(PW8)+presence BCE, one complete 96-window forward."""
    xb, yd, yp, gamma, contrast = batch
    assert chunk == 96, 'PUBLIC_RECIPE_FM_FORWARD_IS_ONE_BATCH_OF96'
    assert len(xb) == 96 and yd.shape == (96, 11, 11, 11) and yp.shape == (96, 11)
    assert xb.shape == (96, 3, 352, 352)
    devices = [torch.cuda.current_device()] if str(device).startswith('cuda') else []
    dense_value = presence_value = 0.
    with torch.random.fork_rng(devices=devices):
        assert 1 <= epoch <= 16 and step >= 1
        fs = fm_fork_seed(seed_base, rng_key, epoch-1, step-1)
        torch.random.default_generator.manual_seed(fs)
        if devices:torch.cuda.manual_seed(fs)
        for start in range(0, len(xb), chunk):
            stop = min(start+chunk, len(xb)); fraction = (stop-start)/len(xb)
            images = fm_photometric(xb[start:stop].to(device), gamma[start:stop].to(device), contrast[start:stop].to(device))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bool(devices)):
                dense = model.fm_dense(images)
            assert dense.shape == (stop-start, 11, 11, 11)
            dense_loss = F.binary_cross_entropy_with_logits(dense.float(), yd[start:stop].to(device),
                                                             pos_weight=torch.tensor(8., device=device))
            presence = dense.float().flatten(2).logsumexp(-1)-math.log(121)
            presence_loss = F.binary_cross_entropy_with_logits(presence, yp[start:stop].to(device))
            loss = coefficient*fraction*(dense_loss+presence_loss)
            assert bool(torch.isfinite(loss)), 'NONFINITE_FM_LOSS'
            loss.backward()
            dense_value += fraction*float(dense_loss.detach())
            presence_value += fraction*float(presence_loss.detach())
    return {'fm_dense': dense_value, 'fm_presence': presence_value,
            'fastmri': coefficient*(dense_value+presence_value)}
