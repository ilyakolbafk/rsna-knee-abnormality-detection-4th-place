"""Opt-in central native-neighbor inputs; reuse baseline transforms and labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from .constants import ID_COLUMN
from .full_source_cache import FullSourceCacheIndex
from .central_sampling import select_centers, source_triplets
from .sharded_dataset import ShardedGrayscaleStudyDataset


def full_source_model_inputs(arrays, sample, image_size):
    if not 0 < image_size <= 336:
        raise ValueError('Full-source bank supports downsampling only')
    refs = sample['bank_refs']
    used = np.unique(refs[refs >= 0])
    # Decompress uint8 once; convert and resize only referenced unique planes.
    planes = torch.from_numpy(arrays['images'][used]).float().div_(255.)
    if image_size != 336:
        planes = F.interpolate(planes[:, None], size=(image_size, image_size),
                               mode='bilinear', align_corners=False, antialias=True)[:, 0]
    tokens = torch.zeros((len(refs), 3, image_size, image_size), dtype=torch.float32)
    available = refs >= 0
    tokens[torch.from_numpy(available)] = planes[torch.from_numpy(np.searchsorted(used, refs[available]))]
    return tokens, sample['token_mask'].copy(), sample['slot_ids'].copy()


class FullSourceStudyDataset(ShardedGrayscaleStudyDataset):
    def __init__(self, manifest, config, *, training, include_labels=True):
        self.manifest = manifest.reset_index(drop=True).copy()
        if ID_COLUMN not in self.manifest or self.manifest[ID_COLUMN].duplicated().any():
            raise ValueError('Manifest must be unique by StudyInstanceUID')
        self.config = config; self.data_config = config['data']; self.train_config = config.get('train', {})
        self.training = training; self.include_labels = include_labels
        self.center_sampling = self.train_config.get('center_sampling', 'fixed')
        if self.center_sampling not in ('fixed', 'central_stratified_v1'):
            raise ValueError('Unknown training center sampling mode')
        if self.center_sampling != 'fixed':
            if self.train_config.get('data_rng_mode') != 'study_epoch_v1':
                raise ValueError('Dynamic centers require reproducible study_epoch_v1 streams')
            if self.data_config.get('center_policy', 'central_uniform') != 'central_uniform':
                raise ValueError('Dynamic training centers require fixed central_uniform evaluation')
        if self.data_config.get('channel_policy') != 'native_source_neighbors':
            raise ValueError('Explicit native_source_neighbors policy required')
        self.cache = FullSourceCacheIndex(self.data_config['cache_index'],
                require_complete=self.data_config.get('require_complete_cache', True),
                expected_studies=int(self.data_config.get('expected_cache_studies', 4407)))
        frozen = self.cache.index['build_contract']
        if self.data_config.get('centering_mode', 'geometric') != 'geometric':
            raise ValueError('Frozen bank uses geometric planar crops')
        for key, value in {'source_image_size':frozen['source_image_size'], 'crop_mm':frozen['crop_mm'],
                           'percentile_low':frozen['old_cache_spec']['percentile_low'],
                           'percentile_high':frozen['old_cache_spec']['percentile_high']}.items():
            if float(self.data_config.get(key, value)) != float(value):
                raise ValueError(f'Configured {key} differs from frozen pixel preprocessing')
        missing = [str(uid) for uid in self.manifest[ID_COLUMN] if str(uid) not in self.cache]
        if missing:
            raise ValueError(f'Full-source bank missing {len(missing)} manifest studies')
        self.image_size = int(self.data_config['image_size'])
        self.token_count = int(self.data_config['token_budget'])
        if int(self.train_config.get('max_train_tokens', self.token_count)) != self.token_count:
            raise ValueError('Do not apply a second token subsampling budget to central source centers')
        if not 0 < self.image_size <= 336 or self.token_count < len(self.data_config['slots']):
            raise ValueError('Invalid resolution or token budget')
        self.anatomy = self.data_config.get('anatomy_features', {})
        self.side_records = None
        if any(self.anatomy.get(k) for k in ('canonical_side', 'position', 'fat_suppression')):
            records = json.loads(Path(self.data_config['side_manifest']).read_text())
            if records['bank_content_sha256'] != self.cache.index['content_sha256']:
                raise ValueError('Side manifest belongs to a different bank')
            self.side_records = records['studies']
            if not set(self.manifest[ID_COLUMN]).issubset(self.side_records):
                raise ValueError('Side manifest missing study')
            want_features = bool(self.anatomy.get('position') or self.anatomy.get('fat_suppression'))
            if bool(config['model'].get('token_metadata', False)) != want_features:
                raise ValueError('Dataset and model disagree on token metadata')

    def _subsample_tokens(self, token_mask, slot_ids):
        return token_mask  # selection has already set the exact budget; consumes no extra RNG

    def sample(self, study_uid):
        arrays, metadata = self.cache.load_study(study_uid)
        expected = [(s['name'], s['plane'], s['fluid_sensitive']) for s in self.data_config['slots']]
        actual = [(s['name'], s['plane'], s['fluid_sensitive']) for s in metadata['legacy_slots']]
        if actual != expected:
            raise ValueError('Full-source semantic slots differ from training configuration')
        if float(metadata['binding']['crop_mm']) != float(self.data_config['crop_mm']):
            raise ValueError('Configured crop differs from materialized source pixels')
        policy = self.data_config.get('center_policy', 'central_uniform')
        rng = None
        if self.training and self.center_sampling == 'central_stratified_v1':
            # EpochSeededDataset sets initial_seed per study/epoch and restores the
            # caller RNG. A separate generator leaves crop/intensity RNG unchanged.
            seed_bytes = f'native64_centers_v1:{torch.initial_seed()}:{study_uid}'.encode()
            seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            policy = 'central_stratified'
        grid_phase = self.data_config.get('inference_grid_phase') if not self.training else None
        if grid_phase is not None:
            policy = 'central_grid'
        centers, slots, selection = select_centers(arrays, metadata, budget=self.token_count,
                trim_each_side=self.data_config.get('trim_each_side', .1),
                slot_weights=self.data_config.get('slot_weights'),
                policy=policy, rng=rng, grid_phase=grid_phase)
        sample = source_triplets(arrays, metadata, centers, slots)
        sample.update(center_refs=centers, selection=selection)
        return arrays, metadata, sample

    def _tokens(self, study_uid):
        arrays, metadata, sample = self.sample(study_uid)
        tokens, mask, slots = full_source_model_inputs(arrays, sample, self.image_size)
        if self.side_records is None:
            return tokens, mask, slots
        from .canonical_inputs import adapt_inputs
        tokens, features, _ = adapt_inputs(tokens, arrays, metadata, sample,
                                           self.anatomy, self.side_records[study_uid])
        if self.config['model'].get('token_metadata', False):
            return tokens, mask, slots, features
        return tokens, mask, slots
