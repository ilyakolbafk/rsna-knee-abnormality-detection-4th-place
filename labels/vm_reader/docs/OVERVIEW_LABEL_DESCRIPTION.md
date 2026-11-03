# Kaggle discussion 733343 — "Knee Abnormality Detection AI Challenge Overview" (Po-Hao "Howard" Chen, competition host; courtesy of Dr. Jacob Kazam). Section "Label Description", verbatim, retrieved 2026-09-18 via browser.

Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific finding in the imaged knee. The labels, and the criteria used by the annotating radiologists, are summarized below. In each case, ambiguous or borderline findings ("on the fence") were graded as negative to favor specificity.

ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee.

# Also from "Imaging Overview" (same post):
"...a meniscal tear, for instance, appears as abnormally increased signal that reaches the surface of the meniscus on more than one image."

# Kaggle discussion 733826 — host reply (Po-Hao "Howard" Chen), verbatim excerpts:
Q: Were the labels assigned independently from the MRI images, rather than extracted from the reports? — "Yes"
Q: If image interpretation and report text disagree, should the image-derived label be considered authoritative? — "Yes. Note that only a small sample of provided data contains both. It is intended to help participants surface this conclusion."
Q: Do negative labels mean that the finding was confirmed absent, or can they also mean that it was not annotated? — "It means the finding was annotated as absent using methods outlined in Overview discussion post under Label Description section."
"Discrepancies are plausible and expected because clinical reports typically involve one signing radiologist who created it for clinical care, and the image-based labels uses multiple readers with stricter image-based thresholds."
