# H6 fastMRI cache (auxiliary task)

H6 adds an auxiliary box task on fastMRI+ (Zhao et al. 2022) to its training step: 96 windows per step from the
1,172 reviewed knee volumes, three views per slice (full field, medial and lateral 100 mm), fastMRI+ boxes mapped to
11 classes. The head is dropped at inference.

Source data: the official NYU fastMRI knee single-coil train and validation archives (free registration at
fastmri.med.nyu.edu) and the public fastMRI+ annotation CSV.

| Folder / file | Role |
|---|---|
| `fm384_v1_build/source_collect.py`, `source_indexed.py`, `fm_indexed_source.py`, `fm_xz_blocks.py` | stream the official archives, keep `reconstruction_rss` and the metadata of each volume |
| `fm384_v1_build/build_candidate.py`, `fm_pixels.py`, `fm_geometry.py`, `finish_pipeline.py` | render 384 px views with the boxes (image flipped vertically once, as in fastMRI+), verify, QA |
| `fm384_anatomy/` (`infer_joint_anchors.py`, `build_anatomical_store.py`, `promote_anatomical_store.py`, ...) | knee anchors that place the medial and lateral views; final pixel store |
| `fm_chunk_loader.py`, `fm_aux_chunk_adapter.py`, `fm_training_reader.py`, `qa_fm_consumer.py` | reader used by the H6 trainer, consumer QA |

The build ran on a rented GPU machine (`deploy.py`, `launch_*.py`, `remote.py` start and watch the remote jobs).
