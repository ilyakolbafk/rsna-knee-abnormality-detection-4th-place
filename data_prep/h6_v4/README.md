# H6 v4 caches

H6 trains on three caches with the same 384 px v4 views (`training/H6_RECIPE.md`):

| Folder | Cache | Source data |
|---|---|---|
| `rsna/` | 4,355 RSNA studies (training set and the three sample test studies) + Gold58 for diagnostics | competition DICOM files |
| `oai/` | 3,254 OAI baseline knees | OAI knee MRI from the NIMH Data Archive |
| `fastmri/` | 1,172 fastMRI knee volumes with fastMRI+ boxes | fastMRI knee single-coil archives (NYU) and the public fastMRI+ annotation CSV |

Each folder has its own README with the commands. Labels are not stored in the caches; the targets are frozen in
`training/sources/h6/assets/{core6,vm}_targets.npz`.

## INPUTS.json

`python train.py --model h6_core6 | h6_vm --h6-inputs INPUTS.json` reads the caches through one JSON file written on the
training machine after the caches are in place (schema `h6_full_v4_inputs_v1`):

- `corpora.RSNA`, `corpora.OAI`: cache descriptors (`backend`, `profile: full_v4`, `root`, `files` with size, mtime,
  ctime and sha256 of the metadata files, `studies` mapping each cache key to its pixel file and metadata);
- `fastMRI`: descriptor of the fastMRI store (`root`, `files` including `index.json`, `boxes.json`, `manifest.json`);
- `studies`: the fit roster (`id`, `corpus`, `source_id` 0 = RSNA / 1 = OAI, `cache_key`, `parent_id`, `report_group`),
  exactly the 7,609 sample ids of the frozen targets;
- `Gold_excluded_ids`: the 58 Gold58 ids (they must not appear in the roster);
- `acceptance`: the manifest sha256 of each cache.

The loaders (`training/sources/h6/full_cache_loader.py`, `fm_data.py`, `main_data.py`) check every file against its
descriptor at run time, so the file has to be written for the machine that trains. We do not ship a generator for it;
the historical file referred to our rented machine and cannot be reused.
