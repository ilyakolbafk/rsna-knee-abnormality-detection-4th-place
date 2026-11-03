# H6 OAI cache

Access: OAI is distributed by the NIMH Data Archive (nda.nih.gov) under a Data Use Certification. Apply for OAI
access with your own NDA account, create a data package with the OAI baseline knee MRI, and set `NDA_USERNAME` and
`NDA_PACKAGE_ID` before running `fetch_case.py` (credentials stay in the NDA keyring). The case roster that maps our
case ids to NDA objects and the OAI targets are not distributed; build your own roster from your data package.

OAI baseline knee MRI (NIMH Data Archive; an NDA account with approved OAI access is required) converted to the
same v4 views as the RSNA cache, so that H6 can train on both. The targets of the OAI knees (MOAKS grades for
Lateral OA, PF OA and synovitis; VM labels for the separate OAI head) are frozen in `training/sources/h6/assets`.

Steps: `fetch_case.py` (download of one case with the NDA download manager), `build_case.py` and `geometry_build.py`
(decode, v8.3 orientation, knee anchors, views), `finalize.py`, QA (`canary_qa.py`, `geometry_qa.py`,
`loader_smoke.py`), reader for the trainer (`reader.py`, `training_reader.py`, `collection_reader.py`).

The historical run streamed finished cases from a workstation to the GPU machine (`pipeline.py`, `upload_case.py`,
`receiver_service.py`); set `H6_CACHE_SSH_HOST`, `H6_CACHE_SSH_PORT` and `H6_CACHE_SSH_KEY` for your own host.
