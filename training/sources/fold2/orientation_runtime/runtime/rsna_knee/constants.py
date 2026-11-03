"""Shared schema constants."""

ID_COLUMN = "StudyInstanceUID"
SERIES_COLUMN = "SeriesInstanceUID"
REPORT_COLUMN = "Report"

TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SERIES_REQUIRED_COLUMNS = [
    ID_COLUMN,
    SERIES_COLUMN,
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]

