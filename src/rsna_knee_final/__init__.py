"""RSNA Knee Abnormality Detection - FINAL ens (Sixway + H6) host package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
