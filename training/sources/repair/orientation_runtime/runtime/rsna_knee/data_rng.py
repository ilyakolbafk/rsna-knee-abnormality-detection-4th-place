"""Study/epoch augmentation streams independent of model construction and workers."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def stream_seed(seed: int, epoch: int, key: str) -> int:
    payload = f'study_epoch_v1:{seed}:{epoch}:{key}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63 - 1)


class EpochSeededDataset(Dataset):
    """Restore caller RNG after each item, including num_workers=0.

    Workers are recreated each epoch (persistent_workers must remain false).
    Study keys must be in exactly the wrapped dataset's positional order.
    """

    def __init__(self, dataset, study_keys, seed: int):
        self.dataset = dataset
        self.study_keys = list(map(str, study_keys))
        self.seed, self.epoch = int(seed), 0
        if len(self.study_keys) != len(dataset) or len(set(self.study_keys)) != len(dataset):
            raise ValueError('RNG study keys must be unique and match dataset length')

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        seed = stream_seed(self.seed, self.epoch, self.study_keys[index])
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                # Data transforms are CPU operations; do not touch accelerator RNG.
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
