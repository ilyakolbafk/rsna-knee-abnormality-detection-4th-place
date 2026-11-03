# Labels

Only 58 training studies come with image-based labels (Gold58, from the official `train.csv`). Every other training
label was produced by us; the tables are part of the frozen training packages (`training/sources/<package>/assets/`),
so a retrained model reads exactly the file its historical run read (sha256 in `training/sources/SOURCE_TREES.json`).

| Table | Content | Used by | Location |
|---|---|---|---|
| `V_MAIN.csv` | **VM labels**: ChatGPT Pro image-reader `subjective_probability` for the 12 findings, 4,352 knee rows from 4,345 studies (7 bilateral studies give one row per knee); `label_mask__*`, `confidence__*` per finding, `fold`, `report_group` | K24 folds, all full fits | `core/assets/V_MAIN.csv` (same file in the fold packages) |
| `FULLFIT_MAIN.csv` | `V_MAIN` + the 58 Gold58 studies with their binary expert labels (4,410 rows) | core, mirror, adaptive | `core/assets/` |
| `NOE1_FULLFIT_MAIN_4413.csv` | `FULLFIT_MAIN` + the 3 public sample test studies read by the VM reader | adaptive | `adaptive/assets/` |
| repaired `V_MAIN.csv` / `FULLFIT_MAIN.csv` | VM labels raised to the core's calibrated out-of-fold prediction where the core is ≥ 0.4 above VM **and** the report readers or a report-trained CNN agree (1.7 % of cells) | repair | `repair/assets/` |
| `E_MAIN.csv` | Evidence-v1: second image read of the same 4,352 knee rows with an evidence-first prompt (`EVIDENCE_MAPPING.csv` maps every row to its accepted record; both knees of one bilateral study and 9 single cells are masked) | joint36 | `joint/assets/` |
| `CORE6_MAIN.csv` | report readers: equal mean of six LLM readings of the report text (two of them are the public label tables of laymond and yunus) | joint36, H6 core6 arm | `joint/assets/` |
| `R_MAIN.csv` | report-derived labels of the earlier pipeline (Core5new); the frozen per-finding `pos_weight` of the CoAtNet recipe was computed on them | class weights (frozen), comparisons | `core/assets/` |
| aux targets (`aux_names.json`, 48 names) | report-derived auxiliary targets (meniscus tear location, cartilage grade ≥ 2 per surface, marrow oedema by zone, effusion size, bursitis, ...), loss weight 0.25 | all CoAtNet models | inside the label tables |
| `core6_targets.npz`, `vm_targets.npz` | H6 targets: RSNA Core6 or VM labels (RSNA rows only; the OAI targets used in the fit are not distributed, see `training/H6_RECIPE.md`) | H6 | `h6/assets/` |
| `gold58/gold58_labels.csv` | the 58 image-labelled training studies (official labels) and their series table | Gold58 diagnostics, full fits | `labels/gold58/` |

How the VM labels were made (rendered panels, reading protocol, output format, validation) is described in
`docs/SOLUTION_DETAILED.md` § 2; `vm_reader/render_reader_panels.py` re-creates the reader input for one study.

## How the tables were made (code and prompts in this folder)

| Folder | What | Key files |
|---|---|---|
| `vm_reader/` | VM image reader ("V62MB"): reader packets rendered on Kaggle from the training DICOM (one CPU notebook per ≈ 600 studies), 30 studies per ordinary ChatGPT Pro chat (4 ZIP bundles + one prompt), one JSON answer per chat, validated and collected per StudyInstanceUID | `build_shard.py` (packet renderer, Kaggle notebook; `kernel-metadata.example.json`), `build_batch_prompts.py`, `PROMPT_EXAMPLE_S01B01.md` (all prompts differ only in case ids), `docs/` (the six shared documents inside every bundle: organiser definitions, clinical checklist v6.2, rules for using the report, required regions, 28 aux field definitions), `render_menu_b_v5_gold58.py` (same packets for Gold58), `render_reader_panels.py` (reader input of one study) |
| `evidence_v1/` | second image read with an evidence-first prompt and the same packets | `PROMPT_EXAMPLE_FEV_R01B01.md`, `prepare_launch.py` |
| `vm_repair/` | consensus repair of the VM labels (rule in `docs/SOLUTION_DETAILED.md` § 2) | `build_vm_repair.py`, `CHANGES.csv` (every changed cell), `RECEIPT.json` |
| `core6/` | the six report readers: prompt of our two text readers, combination into Core6, transfer to all 4,407 studies | `PROMPT_CORE6_TEXT_V1.md`, `build_master.py`, `transfer_4407.py`, `materialize_recipes.py`, `data/train4407_core6_long.csv` (every reader's probability per study and finding) |

The LLM readings themselves cannot be regenerated bit for bit (re-reading the same study flips 4–5 of every 100 binary
cells); the tables above are the exact training inputs.
