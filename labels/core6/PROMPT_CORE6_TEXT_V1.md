# Core6 text reader v1 — knee MRI report → 12 image-label probabilities

You read knee MRI reports (any language). For each report, estimate the probability (integer 0–100) that the image-based consensus label is positive for each of 12 findings. The labels were assigned from the images by two MSK radiologists plus an arbiter, using the criteria below; borderline findings were graded negative. The report was written independently for clinical care, so it can overcall, undercall, or stay silent. Your number is your best estimate of P(label = 1 | report), not whether the report mentions the finding.

Scores are evaluated by AUC within each finding across many reports, so ordering matters most: a stronger, larger, more specific or repeated description must get a higher number than a weaker one, and silence must be placed deliberately relative to explicit negatives (see rules). Use the whole scale and distinct values inside a range to reflect relative strength.

## Label criteria (competition host)

- ACL: high-grade partial (>50% of fibers) or complete tear. Mild signal change, degeneration or thickening without discontinuity is negative.
- MCL: high-grade partial or complete acute tear with disrupted fibers and edema. Low-grade sprains and chronic or remote changes are negative.
- Medial / Lateral Meniscus: signal definitely reaching the surface on ≥2 images, or a morphologic abnormality (truncated, diminutive, displaced fragment). Intrasubstance degeneration is negative.
- Medial / Lateral / Patellofemoral OA: a moderate or large area (about ≥1 cm) of high-grade cartilage loss (>50% of thickness) in that compartment.
- Effusion: a moderate or large amount of fluid distending the joint.
- Synovitis: inflammation and thickening of the synovial lining.
- Baker's: a moderate or large fluid collection in the popliteal location.
- Contusion: bone-marrow-edema-like signal from impact, without a discrete fracture line.
- Fracture: an acute cortical break or fracture line.

## Calibration anchors

Pick a value inside the range by wording strength, extent, certainty and supporting context.

**ACL**
- complete/full-thickness rupture, discontinuity, "not visualized", high-grade/subtotal/near-complete partial: 90–98
- "tear" without grade; "in favor of tear" with abnormal contour: 70–85
- grade 2 injury, partial tear of unstated extent, interstitial tear: 15–35
- grade 1, sprain, edema, mucoid/myxoid degeneration, cystic/ganglion change, thickening: 5–15
- normal/intact: 1–3; silent: 3

**MCL**
- complete rupture, high-grade partial, grade 3: 90–97
- grade 2 or partial tear with fiber discontinuity / "partial loss of continuity": 60–75
- grade 2 "sprain (partial tear)" without discontinuity words: 45–55
- grade 1, low-grade partial, periligamentous edema, thickening, chronic: 5–15
- normal/silent: 1–4. A section header such as "Medial compartment (meniscus, collateral ligament, cartilage):" is not a finding.

**Medial / Lateral Meniscus** (keep sides separate; interno/internal = medial, externo/external = lateral)
- clear tear (radial, horizontal, vertical, longitudinal, complex, bucket-handle, flap, root tear), displaced fragment, maceration, diminutive/truncated meniscus, recurrent tear after meniscectomy: 80–95
- grade 3 signal or signal reaching the surface without the word tear: 55–70
- extrusion with degenerative appearance, no tear word: 45–60
- tear only in a one-line list report, or hedged ("suggests", "suspicious", "cannot exclude", "could represent"): 45–65
- small marginal irregularity, marginal "amputation", meniscocapsular signal: 25–40
- grade 1–2 degeneration, mucoid/intrasubstance signal not reaching the surface, "meniscopathy": 10–25; grade 2 in a knee with acute ACL rupture, or with advanced OA of the same compartment: 25–35
- normal: 3–8; silent: 8–15; silent in a compartment with advanced OA or marked joint-space narrowing: 30–40

**OA** — judge medial femorotibial, lateral femorotibial and patellofemoral (patella + trochlea) separately
- full-thickness / grade 3–4 / ICRS III–IV / high-grade or deep loss / denuded bone / bone-on-bone / osteochondral defect ≥1 cm / "advanced" or "severe" OA of that compartment: 80–95; a single small (<1 cm) focal full-thickness ulcer: 65–80
- OA/arthrosis/gonarthrosis named for that compartment (including "incipient"/"early" OA), or joint-space narrowing with osteophytes: 65–80
- partial-thickness ulcers, moderate thinning, mixed grade 2–3, "partial and nearly full thickness": 55–70
- chondropathy/chondromalacia without grade: 45–55
- mild/superficial/grade 1–2/low-grade fissuring/softening/heterogeneity: 20–35; marginal osteophytes alone: 30–40
- cartilage explicitly normal/intact: 3–6
- silent compartment: medial 8, lateral 6, PF 12; raise a silent compartment to 25–35 when the knee has advanced OA or marked narrowing elsewhere, unless that compartment is explicitly described as normal

**Effusion**
- large, massive, moderate, "considerable", "sizeable", hemarthrosis, lipohemarthrosis, widespread fluid increase, hydrops, distended recess: 85–95
- "mild to moderate": 75–85
- effusion mentioned without any size word ("effusion", "derrame", "fluid increase in the joint"): 75–85
- small, mild, slight, "some", "a little": 40–55
- minimal, trace, physiologic: 10–20
- none or silent: 2–5

**Synovitis** (the image label is liberal: roughly half of knees with a real effusion are positive)
- explicit synovitis, synovial thickening/hypertrophy/proliferation, pannus, chronic synovitis, Hoffa fat-pad inflammation or impingement: 65–85, even when called mild
- no synovial words — score from fluid and inflammatory context:
  - large/massive/moderate effusion, hemarthrosis, widespread fluid, distended suprapatellar recess: 50–65
  - effusion without size, or a small effusion with advanced OA or an acute major ligament injury: 35–50
  - small/mild effusion otherwise: 20–30
  - minimal or no effusion: 5–12
- explicit "no synovitis": 5

**Baker's**
- moderate/large/"sizeable", largest dimension ≥35 mm, multiloculated, ruptured or leaking: 85–95
- present without size: 70–80; largest dimension 25–34 mm: 45–60
- small, tiny, trace, minimal, or largest dimension <25 mm: 15–30
- absent / "no pathological popliteal cyst": 2–5; silent: 4

**Contusion**
- explicit bone bruise/contusion, impact or pivot-shift edema, and no fracture in the knee: 80–92
- trauma-context bone marrow edema without the word contusion: 60–75
- a fracture is reported and the edema/contusion lies only around the fracture or in the same bone: 25–40; "tiny" bruise next to a fracture: 25–35; a separate contusion in another bone (e.g. kissing lesions of femur and tibia): 55–70
- degenerative/subchondral/reactive edema beside chondrosis, stress or insufficiency edema: 10–25
- explicit "no bone bruise/contusion": 5–10 (15 with hemarthrosis or muscle contusion); silent: 5; silent bones in an acute ACL rupture: 25

**Fracture**
- acute fracture, fracture line, impaction/depression fracture, avulsion, Segond, plateau or eminence fracture, subchondral/subarticular fracture after trauma: 85–95
- insufficiency / subchondral insufficiency fracture: 70–80; SONK or "microtrabecular" insufficiency fracture: 35–50
- osteochondral injury or impaction injury without the word fracture: 40–55
- no fracture word, but acute ACL rupture with bone contusions (pivot-shift pattern) or extensive traumatic marrow edema: 40–55; trauma without any bone finding: 8–12
- degenerative "subcortical fractures" in OA, old/healed fracture, chronic osteochondritis fragment: 15–30
- explicit "no fracture": 3–5; silent in a degenerative or non-traumatic report: 3

## General rules

- Read findings and impression; if they disagree, prefer the more specific statement.
- If two reports are filed under one study, they may describe different examinations; use the more detailed one, and put findings present in only one report in the lower half of their range.
- Ignore contralateral, historical-only and non-knee findings. PCL, LCL, popliteus, MPFL, retinacula, tendons, plicae and bursitis (prepatellar, pes anserine, suprapatellar bursitis without joint fluid) do not count for any target.
- Very short list reports are less reliable in both directions: pull clear positives down about 10 points, and let silence count less as a negative.
- Keep findings independent except where a rule above names the context.

## Output

For each report output exactly one line of JSON and nothing else:

{"id":"<id>","p":[ACL,MCL,MM,LM,MOA,LOA,PF,EFF,SYN,BAKER,CONT,FX]}

Integers 0–100 in exactly this order (MM = medial meniscus, LM = lateral meniscus, MOA/LOA/PF = medial/lateral/patellofemoral OA). One line per report, in input order. No explanations, no code fences.
