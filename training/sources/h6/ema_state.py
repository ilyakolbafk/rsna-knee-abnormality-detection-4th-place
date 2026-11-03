"""EMA and fixed SWA state captured alongside every raw optimizer recovery slot."""
import torch
from recovery_state import freeze


class EMA:
    def __init__(self, model):
        self.values = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.updates = 0
        self.last_decay = None

    @torch.no_grad()
    def update(self, model, scheduler_step):
        decay = min(.999, (1+int(scheduler_step))/(10+int(scheduler_step)))
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert current.keys() == self.values.keys()
        for name, value in self.values.items():
            value.mul_(decay).add_(current[name].detach(), alpha=1-decay)
        self.updates += 1
        self.last_decay = decay

    def state_dict(self):
        return {'values': freeze(self.values), 'updates': self.updates, 'last_decay': self.last_decay}

    def load_state_dict(self, state, model):
        assert state['values'].keys() == self.values.keys()
        assert type(state['updates']) is int and state['updates'] >= 0
        params = dict(model.named_parameters())
        self.values = {n: value.to(params[n].device).clone() for n, value in state['values'].items()}
        for n, value in self.values.items():
            assert value.shape == params[n].shape and value.dtype == params[n].dtype
        self.updates, self.last_decay = state['updates'], state['last_decay']

    def full_model_state(self, model):
        result = freeze(model.state_dict())
        assert self.values.keys() <= result.keys()
        result.update(freeze(self.values))
        return result


def swa_update(accumulator, count, ema_model_state):
    # Floating tensors average; integral buffers retain the latest declared epoch.
    if accumulator is None:
        assert count == 0
        return freeze(ema_model_state), 1
    assert accumulator.keys() == ema_model_state.keys()
    for name, value in ema_model_state.items():
        if value.is_floating_point() or value.is_complex():
            accumulator[name].add_((value-accumulator[name])/(count+1))
        else:
            accumulator[name].copy_(value)
    return accumulator, count+1
