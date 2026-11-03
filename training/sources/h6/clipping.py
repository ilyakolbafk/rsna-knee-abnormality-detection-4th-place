import torch

def fm_clip(model, max_norm):
    """v9.8: ONE clip of the merged gradient over every parameter.  The norm of v9.7's parameters is computed exactly as v9.7's
    clip_grad_norm_(model.parameters()) computes it (same function, same tensors, same order), the auxiliary head's separately, and they
    are combined as sqrt(n_main^2 + n_aux^2) -- the L2 norm over all parameters -- or n_main itself when the head's gradient is 0 (lambda = 0:
    then the clip is v9.7's bit for bit); every gradient (the head's included) is then scaled as clip_grad_norm_ scales it."""
    from torch.nn.utils.clip_grad import _get_total_norm, _clip_grads_with_norm_
    params = list(model.parameters())
    main_g = [p.grad for n, p in model.named_parameters() if not n.startswith("aux_") and p.grad is not None]
    aux_g = [p.grad for n, p in model.named_parameters() if n.startswith("aux_") and p.grad is not None]
    n_main = _get_total_norm(main_g, 2.0, False, None)
    if aux_g:
        n_aux = _get_total_norm(aux_g, 2.0, False, None).to(n_main.device)
        total = torch.where(n_aux == 0, n_main, torch.sqrt(n_main.square() + n_aux.square()))
    else:
        total = n_main
    _clip_grads_with_norm_(params, max_norm, total, None)
    return total
