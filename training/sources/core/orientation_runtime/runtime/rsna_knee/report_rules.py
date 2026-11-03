# Frozen report-rule baseline adapted from the public Kaggle notebook:
# https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee
# Source cells: 4, 6, 7; retrieved 2026-08-29.
# Upstream snapshot SHA-256 (joined cells): b51dd390fcf3fb3fea4791c00c65d64aa2ba9a04f95c457ebbde5d481b7ace83
# Keep provenance when modifying or redistributing this baseline.

"""Report -> twelve graded targets, in nine languages.

v2. Differences from the public lexicon are all coverage: the compartment scoping for
osteoarthritis, the pathology vocabulary for cartilage, an asserted-negative path for
findings a report explicitly clears, and a backoff for synovitis, which most reports
never name.
"""
from __future__ import annotations

import re
import unicodedata

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

_PRE = str.maketrans({
    "ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae",
})


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("­", "")
    text = re.sub(r"[_\-/\\]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+")


def unwrap(text: str) -> str:
    """Rejoin lines that a fixed-width layout broke mid-sentence.

    A large share of this corpus arrives hard-wrapped at some column, so a sentence is
    split across two lines with no punctuation at the break. Splitting on newlines then
    severs the finding from its anatomy - `... y parte de la raiz` / `meniscal posterior
    del menisco lateral con extrusion asociada` puts the tear in one clause and the
    meniscus in the next, and neither clause says anything on its own. A line that does
    not end in sentence punctuation is a continuation, not a statement.
    """
    if not isinstance(text, str):
        return ""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if out and out[-1] and not re.search(r"[.;:!?>*•]$", out[-1]) \
                and len(out[-1].split()) >= 4 and s and not s[:1].isupper():
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return "\n".join(out)


def clauses(text: str):
    """Split into clauses; attach `header:` lines to the value beneath them."""
    norm = normalize(unwrap(text) if FEATURES["unwrap"] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            merged.append(c + " " + raw[i + 1])
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend(p.strip() for p in c.split(",") if len(p.split()) > 2)
    return out


# Each rule below that is not obviously right is behind a flag, so that the notebook can
# turn it off and re-measure rather than assert that it helps.
FEATURES = {
    "unwrap": True,               # rejoin hard-wrapped lines before splitting
    "directional_negation": True,  # scope negation by direction instead of by clause
    "oa_inherit": True,            # unlocalised cartilage statements reach all three
    "graded_pathology": True,      # read numeric grades on a per-structure scale
    "synovitis_backoff": True,     # order the silent majority by inflammatory context
}


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- #
# polarity
# --------------------------------------------------------------------------- #
# Negation is scoped by direction, not by clause. `Subchondral insufficiency fracture at
# the medial tibial plateau without articular surface collapse` contains a negator and
# asserts a fracture: `without` governs what follows it, and the fracture precedes it.
# Reading negation at clause scope turns that sentence into a denial - and it is the
# house style of one of the larger reporting sites here, so the error is systematic
# rather than occasional.
PRE_NEG = _rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b",
    r"\bno evidence\b", r"\bfree of\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se\b",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    r"\bkeine?[nmrs]?\b", r"\bohne\b", r"\bnicht\b", r"\bkein\b",
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b",
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν", r"\bουτε\b",
    r"\bбез\b", r"\bне\b", r"липсва", r"\bняма\b",
)

# Turkish and a few Croatian forms put the negator at the end of the sentence, so these
# govern what precedes them instead.
POST_NEG = _rx(
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b",
    r"gozlenmemekte", r"mevcut degil", r"eslik etmiyor", r"\bizlenmedi\b",
    r"izlenmemistir", r"saptanmamistir", r"gorulmemistir", r"\bnema znakova\b",
    r"bez znakova",
)

NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, r"\bunremarkable\b")

NEG_WINDOW = 90


def _negated(clause: str, start: int, end: int) -> bool:
    """True when a negation trigger governs the span [start, end) of this clause."""
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            # `... intact but with a tear` - a contrast conjunction closes the scope.
            if not re.search(r"\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|"
                             r"ωστοσο|αλλα|но)\b", clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    return False

NORMALITY = _rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"\bwithin normal limits\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b",
    r"\bdoga(l|ll)\b", r"korunmus", r"\bnormaldir\b", r"olagan",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt", r"\bprimjeren",
    r"\bodrzanog kontinuiteta", r"\bodržan",
    r"φυσιολογικ", r"ακεραι", r"δεν παρατηρουνται", r"δεν σημειωνονται",
    r"unauffallig", r"regelrecht", r"\bo\.?b\.?\b",
    r"нормал", r"запазен", r"съхранен", r"\bбез особености\b", r"интактн",
    r"\bgaaf\b", r"\bnormaal\b",
)

# "Negator + abnormality-noun" is how most of these languages assert normality:
# `sin alteraciones`, `ohne Auffalligkeiten`, `geen afwijkingen`, `bez osobitosti`.
# Read literally each one is a negation, so the old guard (`normality AND NOT negation`)
# threw every one of them away and left the structure looking unmentioned instead of
# explicitly clear. That matters to a rank metric: a ligament a radiologist looked at and
# called intact must rank below one the report never mentions, not equal to it.
NORMAL_PHRASE = _rx(
    r"\bsin alteracion", r"\bsin cambios\b", r"\bsin particularidad",
    r"\bsin hallazgos\b", r"\bsin lesion", r"\bsin signos de (rotura|lesion)",
    r"\bcontinu[oa]s?\b", r"\bcontinuidad conservada\b",
    r"\bno abnormalit", r"\bno significant abnormalit", r"\bunremarkable\b",
    r"\bno evidence of (tear|injury|abnormalit)",
    r"\bohne auffalligkeit", r"\bkein nachweis\b", r"\bohne befund\b",
    r"\bgeen afwijking", r"\bzonder afwijking",
    r"\bsans anomalie", r"\bpas d[e']anomalie",
    r"\bbez osobitosti\b", r"\bbez znakova (rupture|lezije)\b",
    r"\bbez patoloskih\b",
    r"χωρις αλλοιωσ", r"χωρις παθολογ", r"δεν παρατηρουνται (αξιολογα|παθολογ)",
    r"\bбез особености\b", r"\bбез патологич", r"\bбез данни за\b",
    r"\bozel bir ozellik yok", r"\bpatolojik bulgu (yok|izlenmemis)",
)

UNCERTAIN = _rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected?\b",
    r"cannot (be )?exclude", r"\bmay\b", r"\bquestionable\b", r"\bequivocal\b",
    r"\br/o\b", r"\bdd\b", r"\blikely\b", r"\bsuggest", r"\bcompatible with\b",
    r"\bposible\b", r"sin criterios categoricos", r"\bdudos", r"\bsugier",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim", r"\bdusundur",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b", r"\bmoze odgovarati\b",
    r"πιθαν", r"υποπτ",
    r"\bmoglich", r"\bverdachtig", r"\bfraglich", r"\bv\.?a\.?\b", r"\bwohl\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект",
    r"\bmogelijk\b", r"\bverdacht\b",
)

# --------------------------------------------------------------------------- #
# pathology vocabulary
# --------------------------------------------------------------------------- #
TEAR = _rx(
    r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
    r"\bavuls", r"\bmacerat", r"\bbuckethandle\b", r"bucket handle",
    r"\brotura\b", r"\broturas\b", r"\bruptura", r"\bdesgarro", r"\broto\b",
    r"\bdechirure", r"\bdechire",
    r"\bscheur", r"\bruptuur", r"gescheurd",
    r"\briss\b", r"einriss", r"\bruptur", r"zerreiss", r"\blasion", r"\bausriss",
    r"\byirtik", r"\byirtig", r"\bkopma\b", r"butunluk kaybi", r"\brupturu\b",
    r"devamsizlik", r"\brupture\b", r"\bdevamliligi secilememis",
    r"\bpuknuce", r"\bprekid\b", r"\bpukotin", r"\bruptur",
    r"ρηξη", r"ρηξις", r"ρηγμα", r"ασυνεχεια",
    r"руптура", r"разкъсв", r"разрив", r"скъсв", r"\bлезия\b",
)

DEGEN = _rx(
    r"degenerat", r"\bmucoid\b", r"\bmyxoid\b", r"\bfray", r"\bfissur",
    r"dejeneratif", r"\bmukoid\b", r"degenerativn", r"εκφυλ", r"дегенерат",
    r"\bμυξοειδ", r"\bμυξωδ", r"\bmeniskopat", r"\bmeniscopath",
    r"\bmuco ?ide\b", r"aufgefasert", r"\bdejenerasyon\b",
)

INJURY = _rx(
    r"\binjur", r"\bsprain", r"\blesion", r"\blasion", r"\bedema\b", r"\boedema\b",
    r"\bodem\b", r"\bedem\b", r"\bοιδημα", r"\bодем", r"\bедем", r"\bstrain\b",
    r"\bhigh signal\b", r"\bsignal alteration\b", r"\bhiperintens", r"\bhyperintens",
    r"aumento de senal", r"alteracion de senal", r"cambio de senal",
    r"\bsignalanhebung", r"\bsignalalteration", r"verhoogd signaal", r"sinyal artis",
    r"αυξημενο σημα", r"повишен сигнал", r"\besguince\b",
    r"\bthicken", r"\bzadebljanje\b", r"\bverdikking\b", r"\bdistenzij",
    r"\blaksite\b", r"\blaxity\b", r"\bpartial\b", r"\bparcijaln", r"\bparcial",
    r"\bpartiel", r"\bpartiell",
)

# A numeric grade is the most precise thing a knee report says, and it means different
# things in different places: grade 3 of a meniscus is a tear by definition, grade 1 or 2
# is intrasubstance signal that never reaches the surface and is not one. A ligament runs
# the other way round - grade 1 is a stretch, grade 2 a partial tear. So the grade is
# read as a number and interpreted per structure rather than folded into one pathology
# vocabulary.
_GRADE_RX = re.compile(
    r"(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)"
    r"[\s:]*(?:grade\s*)?([1-4]|iv|iii|ii|i)\b"
)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def _grade_of(clause: str):
    """Highest numeric grade stated in a clause, or None."""
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best

# --------------------------------------------------------------------------- #
# anatomy
# --------------------------------------------------------------------------- #
ANAT = {
    "ACL": _rx(
        r"anterior cruciate", r"\bacl\b",
        r"cruzado anterior", r"\blca\b",
        r"croise anterieur",
        r"voorste kruisband", r"\bvkb\b",
        r"vorderes kreuzband", r"vorderen kreuzband", r"vordere kreuzband",
        r"on capraz", r"\bocb\b", r"anterior capraz",
        r"prednji krizni", r"prednjeg krizn",
        r"προσθι[οα][^ ]* χιαστ", r"προσθιου χιαστου", r"χιαστο[^ ]* συνδεσμ",
        r"\bχιαστ\w*",
        r"предна кръстна", r"предната кръстна", r"предна кръста",
        r"cruciate ligaments", r"ligamentos cruzados", r"ligaments croises",
        r"kruisbanden", r"kreuzbander", r"capraz baglar", r"krizn[a-z]* ligament[a-z]*",
        r"χιαστοι συνδεσμ", r"χιαστων συνδεσμ", r"кръстните връзки", r"кръстни връзки",
    ),
    "MCL": _rx(
        r"medial collateral", r"\bmcl\b", r"tibial collateral",
        r"colateral medial", r"colateral interno", r"\blcm\b",
        r"collateral medial", r"collateral interne",
        r"mediale collaterale", r"binnenband", r"\b(mediale|laterale) banden\b",
        r"\bcollaterale banden\b",
        r"innenband", r"mediales? kollateral",
        r"\bic yan bag", r"medial kollateral", r"\biyb\b", r"medyal kollateral",
        r"medijalni kolateraln", r"medijalnog kolateraln",
        r"εσω πλαγι", r"εσωτερικο πλαγι", r"\bπλαγι\w* συνδεσμ", r"\bπλαγιοι\b",
        r"медиален колатерал", r"вътрешна странична", r"\bколатерал\w*",
        r"\bcolaterales\b", r"\bcollateraux\b", r"\bcollateralen\b", r"\bkolateralni\b",
        r"collateral ligaments", r"ligamentos colaterales", r"ligaments collateraux",
        r"collaterale banden", r"kollateralbander", r"seitenbander", r"yan baglar",
        r"kolateraln[a-z]* ligament[a-z]*", r"πλαγιοι συνδεσμ", r"πλαγιων συνδεσμ",
        r"колатерални връзки", r"страничните връзки",
    ),
    "Medial Meniscus": _rx(
        r"medial meniscus", r"\bmm\b(?= tear)", r"medial menisc",
        r"menisco medial", r"menisco interno",
        r"menisque medial", r"menisque interne",
        r"mediale meniscus", r"binnenmeniscus",
        r"innenmeniskus", r"medialen? meniskus", r"innenmeniskushinterhorn",
        r"medyal menisk", r"\bic menisk",
        r"medijalni meniskus", r"medijalnog meniskusa", r"medijalnom meniskusu",
        r"medijaln\w* menisk\w*", r"\bmedijalnog meniska\b", r"medijalni menisk",
        r"εσω μηνισκ", r"μηνισκ[^ ]* του εσω", r"εσω διαμερισμα[^.]{0,40}μηνισκ",
        r"медиалния менискус", r"медиален менискус", r"вътрешния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"amfoteroi\w* mhnisk", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
    "Lateral Meniscus": _rx(
        r"lateral meniscus", r"lateral menisc",
        r"menisco lateral", r"menisco externo",
        r"menisque lateral", r"menisque externe",
        r"laterale meniscus", r"buitenmeniscus",
        r"aussenmeniskus", r"lateralen? meniskus", r"aussenmeniskushinterhorn",
        r"lateral menisk", r"\bdis menisk",
        r"lateralni meniskus", r"lateralnog meniskusa", r"lateralnom meniskusu",
        r"lateraln\w* menisk\w*", r"\blateralnog meniska\b",
        r"εξω μηνισκ", r"μηνισκ[^ ]* του εξω", r"εξω διαμερισμα[^.]{0,40}μηνισκ",
        r"латералния менискус", r"латерален менискус", r"външния менискус",
        r"oba meniska", r"both menisci", r"ambos meniscos", r"beide menisci",
        r"her iki menisku", r"αμφοτερ\w* μηνισκ",
        r"двата менискуса", r"medial (and|&) lateral menisc",
    ),
}

# Osteoarthritis is written as cartilage damage far more often than as a diagnosis.
OA_EVIDENCE = _rx(
    r"osteoarthrit", r"\barthros", r"\bgonarthros", r"\bosteoarthros",
    r"chondropath", r"chondromalac", r"condropat", r"condromalac", r"\bchondros",
    r"\bchondrosis\b", r"chondral (loss|defect|ulcer|thinning|injury|fissur|wear)",
    r"cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)",
    r"(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage",
    r"articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)",
    r"osteophyt", r"osteofit", r"osteofyt", r"osteofito", r"osteophyten", r"spurring",
    r"joint space narrowing", r"pinzamiento articular", r"reduced joint space",
    r"kikirdak kayb", r"kikirdak incelme", r"kondropati", r"kondral", r"kikirdak dejener",
    r"eklem aralig\w* daral", r"eklem mesafesi daral", r"kikirdak kalinlig\w* azal",
    r"kraakbeen", r"gonartrose", r"artrose", r"\bknorpel", r"arthrose", r"gonarthrose",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit", r"artrotsk", r"artrotick",
    r"\boa promjen", r"\boa\b", r"degenerativne promjene hrskav",
    r"χονδρ[^ ]*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ", r"χονδρομαλακ",
    r"αρθρικου χονδρου", r"εξαλειψη του αρθρικου χονδρου", r"διαβρωση του αρθρικου χονδρ",
    r"λεπτυνση[^.]{0,30}χονδρ", r"φθορα[^.]{0,20}χονδρ",
    r"артроз", r"хондропат", r"остеофит", r"хрущял[^.]{0,40}(изтън|увред|дефект|липс)",
    r"изтъняване[^.]{0,30}хрущял", r"хондромалац",
    r"ulcera[s]? condral", r"cartilago[^.]{0,25}(perdida|adelgaz)",
    r"icrs grade", r"icrs\b", r"outerbridge", r"\bdenudation\b", r"denudacij",
    r"erozivne promjene", r"\berosion of[^.]{0,20}cartilage",
    r"kraakbeenlijden", r"kraakbeenverlies",
)

# --------------------------------------------------------------------------- #
# where in the joint a cartilage statement sits
# --------------------------------------------------------------------------- #
# Tibiofemoral structures. Used only inside a clause that already carries OA evidence,
# so bare "condyle" is safe here and would not be elsewhere.
TF_SITE = _rx(
    r"compartment", r"compartimento", r"compartiment", r"kompartman", r"kompartiment",
    r"kompartment", r"odjelj", r"διαμερισμα", r"компартм", r"\bотдел",
    r"femorotibial", r"tibiofemoral", r"femoro tibial", r"femorotibiaal",
    r"femorotibijaln", r"феморотибиал", r"\bft zglob", r"tibiofemoraln",
    r"condyle", r"condilo", r"kondyl", r"kondil", r"condyl", r"κονδυλ",
    r"кондил", r"\bplateau", r"\bplato\b", r"platillo", r"meseta", r"плато",
    r"tibiaplateau", r"tibijaln\w* plato", r"tibyal plato", r"tibia plato",
    r"κνημιαι", r"μηριαι", r"weightbearing", r"weightbaring", r"zona de carga",
    r"dragende deel", r"agirlik tasiyan", r"\bfemur\b", r"\btibia\b", r"\bfemoral\b",
    r"\btibial\b", r"\bfemura\b", r"\btibije\b", r"\bmesarthrio\b", r"μεσαρθριο",
)

# Patellofemoral structures.
PF_SITE = _rx(
    r"patellofemoral", r"femoropatellar", r"femoropatelar", r"patelofemoral",
    r"retropatellar", r"retrorotulian", r"trochlea", r"troclea", r"troklea",
    r"trochlear", r"trohlej", r"τροχιλ", r"\bpatella", r"\bpatellar", r"rotulian",
    r"\brotula\b", r"\bpatele\b", r"patellofemoraal", r"femoropatellair",
    r"επιγονατιδ", r"μηροεπιγονατιδ", r"пател", r"феморопател",
    r"anterior compartment", r"compartimento anterior", r"prednj\w* odjeljk",
    r"\bfp zglob", r"\bpf zglob", r"\bfaset", r"\bfacet", r"patellofemoraln",
)

SIDE_MEDIAL = _rx(
    r"\bmedial\w*", r"\bmedyal\w*", r"\bmedijaln\w*", r"\bmediaal\w*",
    r"\bmediale\w*", r"\binterno\b", r"\binterna\b", r"\binternos\b", r"\binterne\b",
    r"\binnen\w*", r"\bic\b", r"\bunutarnj\w*", r"\bεσω\w*", r"\bεσωτερικ\w*",
    r"\bмедиал\w*", r"\bвътреш\w*", r"\bbinnen\w*", r"\bmediaal\b", r"\bmediales?\b",
)
SIDE_LATERAL = _rx(
    r"\blateral\w*", r"\bexterno\b", r"\bexterna\b", r"\bexternos\b", r"\bexterne\b",
    r"\bdis\b", r"\blateraln\w*", r"\baussen\w*", r"\bbuiten\w*", r"\bεξω\w*",
    r"\bεξωτερικ\w*", r"\bлатерал\w*", r"\bвъншн\w*", r"\bvanjsk\w*",
)
SIDE_ANTERIOR = _rx(
    r"\banterior\w*", r"\bant\b", r"\bon\b", r"\bprednj\w*", r"\bvorder\w*",
    r"\bvoorste\b", r"\bπροσθι\w*", r"\bпредн\w*", r"\banteriyor\w*", r"\bavant\b",
    r"\banterieur\w*",
)

GLOBAL_OA = _rx(
    r"tri ?compartment", r"all three compartment", r"global(ised)? (oa|osteoarthrit)",
    r"\bgonarthros", r"\bgonartros", r"\bgonarthrose", r"\bgonartrose", r"gonartro",
    r"goanrtrot", r"gonartrot",
    r"osteoarthritis of the knee", r"artrosis (de |)(la )?rodilla", r"knee osteoarthrit",
    r"\bdiz osteoartrit", r"\bgonartroz", r"artroza koljena",
    r"οστεοαρθριτιδα", r"αρθριτιδα του γονατος", r"εκφυλιστικη οστεοαρθριτ",
    r"артроза на колянната", r"гонартроз",
    r"degenerative joint disease", r"\bdjd\b", r"three compartments",
    r"compartmens", r"compartments",
)

# --------------------------------------------------------------------------- #
# self-declaring findings
# --------------------------------------------------------------------------- #
DIRECT = {
    "Effusion": _rx(
        r"\beffusion", r"joint fluid", r"intra ?articular fluid", r"\bhydrops\b",
        r"\bhemarthros", r"\bhaemarthros",
        r"derrame articular", r"\bderrame\b", r"liquido articular", r"hemartrosis",
        r"epanchement",
        r"gewrichtsvocht", r"\bvocht\b", r"gewrichtseffusie", r"opzetting van suprapatell",
        r"gelenkerguss", r"\berguss\b", r"gelenksergu", r"gelenksflussigkeit",
        r"eklem\w* ic\w* sivi", r"efuzyon", r"eklem sivisi", r"eklem mesafesinde sivi",
        r"sivi (miktari|artisi|birikimi)", r"sivi artis", r"\bsivi\b[^.]{0,25}artmis",
        r"\bizljev", r"\bizliv", r"zglobn[^ ]* tekucin", r"\bhidrops\b",
        r"αρθρικ[^ ]* υγρ", r"υγρου ενδαρθρικα", r"ενδαρθρικ[^ ]* υγρ", r"ποσοτητα υγρου",
        r"ενδαρθρικ", r"αρθρικη συλλογη", r"υγρο στην αρθρωση", r"υγρου στην αρθρωση",
        r"συλλογη υγρου", r"ενθαρθρικ",
        r"ставен излив", r"излив", r"ставна течност", r"синовиална течност",
    ),
    "Synovitis": _rx(
        r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
        r"thicken\w* synovial", r"hypertroph\w* of the synovium",
        r"synoviale? (verdikking|proliferatie)", r"verdikkingen van (het )?synovium",
        r"synovialitis", r"synovialis(verdickung|proliferation)", r"reizsynovial",
        r"sinovijalitis", r"sinovitis", r"zadebljanje sinovij", r"proliferacij\w* sinovij",
        r"sinovijaln\w* proliferacij",
        r"υμενιτιδα", r"συνοβιτιδα", r"υμενικ[^ ]* υπερτροφ", r"αρθρικου υμεν",
        r"παχυνση[^.]{0,20}υμεν", r"υμενα",
        r"синовит", r"синовиал[^ ]* (задебел|пролифер)",
        r"\bpannus\b", r"\bhoffit", r"sinovyal\w* (kalinlas|proliferas)",
        r"sinovyal hipertrof", r"\bartrit\b", r"\barthritis\b",
    ),
    "Baker's": _rx(
        r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
        r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
        r"popliteal kist", r"\bbakerova\b", r"poplitealn[^ ]* cist", r"popliteal\w* cist",
        r"κυστη baker", r"πολυχωρη συνοβιακη κυστη", r"κυστη του baker",
        r"συνοβιακη κυστη", r"κυστη τυπου baker",
        r"киста на бейкър", r"бейкърова киста", r"поплитеална киста", r"бекеров",
        r"gastrocnemio ?semimembranos", r"gastrocnemius semimembranosus burs",
    ),
    "Contusion": _rx(
        r"\bcontusion", r"bone bruise", r"bone marrow (o?edema|contusion)",
        r"marrow o?edema", r"\bkontuz", r"medular bone o?edema", r"osseous contusion",
        r"contusion osea", r"edema oseo", r"edema de medula osea", r"contusiones oseas",
        r"oedeme osseux", r"contusion osseuse",
        r"botcontusie", r"botoedeem", r"beenmergoedeem", r"botmergoedeem",
        r"knochenmarkodem", r"knochenodem", r"knochenmarksodem", r"kontusion",
        r"kemik kontuzyonu", r"kemik iligi odemi", r"kemik odemi", r"kemik iliginde odem",
        r"kontuzyonel kemik", r"kemik iligi odemleri",
        r"kostani edem", r"edem kosti", r"kontuzij", r"kostane srzi[^.]{0,20}edem",
        r"οστεομυελικ[^ ]* οιδημα", r"οστικο οιδημα", r"μυελικο οιδημα", r"οστικο μωλωπ",
        r"костномозъчен едем", r"костен едем", r"контузионен", r"костно мозъчен едем",
    ),
    "Fracture": _rx(
        r"\bfractur", r"\bfract\b",
        r"\bfractura", r"\bfracturas\b",
        r"\bfractuur", r"\bbreuk\b",
        r"\bfraktur", r"\bbruch\b",
        r"\bkirik\b", r"\bkirigi\b", r"\bkiri[kg]\w*",
        r"\bprijelom", r"impresijsk[^ ]* fraktur", r"impaktcij",
        r"καταγμα", r"καταγματ",
        r"фрактур", r"счупван", r"фисур",
        r"insufficiency fracture", r"stress fracture", r"avulsion fracture",
        r"subchondral fracture", r"subkondral kiri", r"impaction (fracture|injury)",
        r"osteochondral (fracture|impaction)", r"\bsegond\b", r"impactiefractuur",
        r"subchondrale impression", r"subchondraler? impress",
    ),
}

DECOY = {
    "Fracture": _rx(r"microfractur", r"\bfracture (risk|prophyla)"),
    "Baker's": _rx(r"meniscal cyst", r"quiste meniscal", r"parameniscal"),
}

PAIRED = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"}
OA_TARGETS = ["Medial OA", "Lateral OA", "PF OA"]

# --------------------------------------------------------------------------- #
# stems, for morphology the phrase lexicons cannot reach
# --------------------------------------------------------------------------- #
# A report that clears or tears both menisci in one breath - "normaal voorkomen
# menisci", "Normal medial and lateral menisci" - names neither side, so a
# side-qualified lexicon leaves both targets silent. The cruciates and collaterals
# already had their plural forms; the menisci were simply missed.
#
# It is only consulted when the clause names no side at all. Otherwise "tear of the
# medial meniscus, menisci otherwise intact" would fire the plural for the lateral side
# off a clause that is about the medial one.
PLURAL_MENISCI = _rx(
    r"\bmenisci\b", r"\bmeniscos\b", r"\bmenisques\b", r"\bmenisken\b",
    r"\bmeniskusi\b", r"\bmenisk\w*ler\b", r"\bμηνισκοι\b", r"\bμηνισκων\b",
    r"\bменискуси\b", r"\bменискусите\b", r"\bmenisci\w*\b",
)
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)

STEM_MENISCUS = _rx(r"menisc\w*", r"menisk\w*", r"μηνισκ\w*", r"мениск\w*")
STEM_CRUCIATE = _rx(r"cruciate", r"cruzado", r"croise", r"kruisband", r"kreuzband",
                    r"capraz bag\w*", r"krizn\w*", r"χιαστ\w*", r"кръстн\w*",
                    r"\bacl\b", r"\blca\b", r"\bvkb\b", r"\bocb\b", r"\bacb\b")
STEM_COLLATERAL = _rx(r"collateral\w*", r"colateral\w*", r"kollateral\w*",
                      r"collaterale\w*", r"kolateraln\w*", r"yan bag\w*",
                      r"πλαγι\w*", r"колатерал\w*", r"странич\w*",
                      r"innenband\w*", r"binnenband\w*", r"\bmcl\b", r"\blcm\b",
                      r"\biyb\b")

STEM_FRACTURE = _rx(r"fractur\w*", r"fraktur\w*", r"fractuur\w*", r"\bfract\b",
                    r"kiri[kgğ]\w*", r"prijelom\w*", r"lom kosti", r"\bbreuk\w*",
                    r"\bbruch\w*", r"καταγμα\w*", r"καταγματ\w*", r"фрактур\w*",
                    r"счупван\w*", r"fisur\w* (osea|oseas|kost)", r"fissur\w* kost")

# Decoys that steal a cruciate / collateral stem match for the wrong ligament.
POSTERIOR_ONLY = _rx(r"\bpcl\b", r"\blcp\b", r"\bhkb\b", r"\bacb\b",
                     r"posterior cruciate", r"cruzado posterior", r"croise posterieur",
                     r"achterste kruisband", r"hinteres kreuzband", r"arka capraz",
                     r"straznji krizn", r"οπισθι[οα]\w* χιαστ", r"задна кръстн",
                     r"задната кръстн")
LATERAL_COLL_ONLY = _rx(r"\blcl\b", r"\bfcl\b", r"lateral collateral",
                        r"fibular collateral", r"colateral lateral", r"colateral externo",
                        r"buitenband", r"aussenband", r"dis yan bag",
                        r"lateralni kolateraln", r"εξω πλαγι", r"латерален колатерал")

def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int = 55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False


STEM_RULES = {
    "ACL": (STEM_CRUCIATE, SIDE_ANTERIOR),
    "MCL": (STEM_COLLATERAL, SIDE_MEDIAL),
    "Medial Meniscus": (STEM_MENISCUS, SIDE_MEDIAL),
    "Lateral Meniscus": (STEM_MENISCUS, SIDE_LATERAL),
}


class _Matcher:
    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None


ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {
    t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == "Fracture" else rx)
    for t, rx in DIRECT.items()
}

# --------------------------------------------------------------------------- #
# severity
# --------------------------------------------------------------------------- #
SEV_LOW = _rx(
    r"\bsmall\b", r"\bminimal\b", r"\btrace\b", r"\bmild\b", r"\bslight\b",
    r"\btiny\b", r"\bscant\b", r"\bdiscrete\b", r"\blow ?grade\b", r"\bincipient\b",
    r"\bleve\b", r"\bminim", r"\bpeque", r"\bfina\b", r"\bfino\b", r"\bligero\b", r"\bescaso\b", r"\bdiscreto\b",
    r"\bhafif\b", r"\baz miktarda\b", r"\bsilik\b",
    r"\bmanj\w*", r"\bblago\b", r"\bdiskretn", r"\bmalo\b", r"\bpocetn",
    r"\bgering", r"\bdiskret", r"\bkleine?r?\b", r"\bwenig\b", r"\bzarte?\b",
    r"\bbeperkte?\b", r"\bgeringe\b", r"\bweinig\b", r"\blichte?\b", r"\blicht\b",
    r"\bηπι", r"\bμικρ", r"\bελαχιστ", r"\bαρχομεν",
    r"\bминимал", r"\bлек", r"\bмалк", r"\bнеголям",
)

SEV_HIGH = _rx(
    r"\blarge\b", r"\bmarked\b", r"\bmassive\b", r"\bsevere\b", r"\bextensive\b",
    r"\bmoderate\b", r"\bgross\b", r"\bsignificant\b", r"\babundant\b", r"\btense\b",
    r"\bcomplete\b", r"\bfull ?thickness\b", r"\bhigh ?grade\b", r"\badvanced\b",
    r"\bmoderad", r"\bimportante\b", r"\bsevera?\b", r"\bmarcad", r"\bcuantios",
    r"\bespesor total\b", r"\bcompleta?\b",
    r"\bbelirgin\b", r"\byaygin\b", r"\bileri\b", r"\bciddi\b", r"\bbol\b", r"\bkomplet",
    r"\bopsezan\b", r"\bveliki\b", r"\bizrazit", r"\bznacajn", r"\bumjeren",
    r"\buznapredoval", r"\bpotpun", r"\bkompleksn",
    r"\bausgepragt", r"\bdeutlich", r"\bmassiv", r"\bmassig", r"\bgross",
    r"\buitgebreid", r"\bgevorderd", r"\bveel\b", r"\bmatige?\b", r"\bvolledig",
    r"\bμετρι", r"\bμεγαλ", r"\bεκτεταμεν", r"\bευμεγεθ", r"\bσοβαρ", r"\bπληρη",
    r"\bголям", r"\bизразен", r"\bзначим", r"\bумерен", r"\bобилен", r"\bпълн",
)

GRADE_HIGH = re.compile(r"grade?[ao]?\s*(3|4|iii|iv)\b|icrs grade (iii|iv|3|4)|"
                        r"stupnja iv|stupnja iii|\bgrado (3|4)\b|\bgrad (3|4)\b|"
                        r"\bgrade (3|4)\b")

DEGENERATIVE_MARROW = _rx(
    r"subchondral", r"subcondral", r"subkondral", r"supkondraln", r"subchondraln",
    r"υποχονδρι", r"υπαρθρικ", r"субхондрал", r"subchondrale?", r"subartikuler",
    r"\bcyst", r"\bquist", r"\bzyste\b", r"\bcistic", r"reactive", r"reactivo",
    r"degenerative", r"degenerativ", r"reaktiv", r"\bcisti\b",
)

TRAUMA = _rx(
    r"\bbruise\b", r"\bcontusion", r"\bkontuz", r"\btrauma", r"\bimpaction\b",
    r"\bpivot shift\b", r"\bkissing\b", r"\bacute\b", r"\bagudo\b", r"\bakut",
    r"\bpivot kaymasi\b", r"\bcontusion osseuse\b", r"\bbone bruise\b",
    r"\bbotcontusie\b", r"\bконтузион", r"\bμωλωπ", r"\bkontuzij", r"\bimpaktcij",
    r"\bimpakcij", r"\bfall\b", r"\binjury\b", r"\bimpression\b",
)

# Effusion-adjacent inflammatory signs, used only when a report never names synovitis.
SYNOVIAL_PROXY = _rx(
    r"bursit", r"burzit", r"\bbursa\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)",
    r"suprapatellar (bursitis|effusion|recess)", r"suprapatellar bursa",
    r"suprapatellar bursada", r"suprapatelarno", r"suprapatellaire recessus",
    r"hoffa", r"hoffit", r"plica", r"plika", r"πλικα", r"fat pad[^.]{0,20}(edema|oedema)",
    r"kapsul", r"capsul", r"καψ", r"капсул", r"\bpannus\b", r"\bsinov", r"\bsynov",
)


def _polarity(clause: str, span=None) -> str:
    """positive / negative / uncertain for a term matched at `span` in this clause."""
    if UNCERTAIN.search(clause):
        return "uncertain"
    if span is None or not FEATURES["directional_negation"]:
        if NEGATION.search(clause):
            return "negative"
    elif _negated(clause, span[0], span[1]):
        return "negative"
    if NORMALITY.search(clause):
        # `meniscus normal` denies; `normal alignment ... full thickness tear` does not.
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return "positive"
        return "negative"
    return "positive"


def _severity(clause: str) -> float:
    """How emphatic a clause is about the finding it asserts.

    A numeric grade is deliberately not read here. It belongs to whichever structure the
    grade was written for, and a clause that grades the cartilage while mentioning the
    effusion in passing - `PF arthrotic change with reduced joint space, a smaller
    effusion and grade IV chondromalacia` - would otherwise report a severe effusion.
    """
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and not low:
        return 1.0
    if low and not high:
        return 0.45
    if high and low:
        return 0.8
    return 0.75


def _grade(n_pos, n_neg, n_unc, best):
    """Map counted evidence onto a score in (0, 1) and a confidence."""
    if n_pos or n_unc:
        score = min(0.97, 0.50 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.20 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = 0.28, 0.05
    return score, conf


def _paired_weight(clause: str, meniscus: bool) -> float:
    """How strongly one clause asserts damage to a meniscus or a cruciate/collateral.

    Ordered, not calibrated. What has to hold is that a tear outranks a graded lesion,
    that the grade is read on the right scale for the structure, and that intrasubstance
    degeneration lands below both - the annotator marks a torn meniscus and leaves a
    degenerate one, and a lexicon that scores the two alike throws that ordering away.
    """
    g = _grade_of(clause) if FEATURES["graded_pathology"] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.30
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    else:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.85 if g >= 2 else 0.30
        elif DEGEN.search(clause):
            base = 0.40
        else:
            base = 0.55
    if SEV_HIGH.search(clause) and not SEV_LOW.search(clause):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and not SEV_HIGH.search(clause):
        base *= 0.7
    return base


def _score_paired(cls, tgt):
    """Evidence for one of the four side-specific ligament / meniscus targets."""
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = "Meniscus" in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) \
                and not ANY_SIDE.search(c):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        # Anchor the negation test on the pathology word, not on the anatomy word.
        # `Medial meniscus is not torn` negates the tear, and the negator stands after
        # the noun: scoping from the noun would read the sentence as an assertion.
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == "positive":
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return s, cf, n_pos, n_neg


def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None,
                   context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and not path_rx.search(c):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c)
                                           and not NEGATION.search(c)):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == "positive":
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == "negative":
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.30)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return s, c, n_pos, n_neg


def _score_oa(cls):
    """Osteoarthritis, scoped to the three compartments.

    A cartilage statement is attributed by what else the clause names rather than by a
    compartment phrase: `medial` next to a tibiofemoral structure sends it to the medial
    compartment, `patella` or `trochlea` to the patellofemoral one, and a statement that
    names neither is a whole-joint assertion that counts for all three at a discount.
    That last case is what the public lexicon leaves silent, and it is the majority of
    the corpus - reports say `tricompartmental chondrosis`, not `chondrosis of the
    medial femorotibial compartment`.
    """
    acc = {t: {"pos": 0, "neg": 0, "unc": 0, "best": 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = 0, 0, 0.0

    for c in cls:
        m = OA_EVIDENCE.search(c)
        if not m:
            continue
        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None
        hits = []
        if tf_med:
            hits.append("Medial OA")
        if tf_lat:
            hits.append("Lateral OA")
        if pf:
            hits.append("PF OA")

        if not hits:
            # No compartment named. Whole-joint statements ("tricompartmental
            # chondrosis", "gonarthrose") speak for all three; an unlocalised cartilage
            # remark is weaker evidence but still evidence, so it is carried at a
            # discount rather than dropped.
            if pol == "positive":
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == "negative":
                g_neg += 1
            continue

        for t in hits:
            if pol == "positive":
                acc[t]["pos"] += 1
                acc[t]["best"] = max(acc[t]["best"], sev)
            elif pol == "negative":
                acc[t]["neg"] += 1
            else:
                acc[t]["unc"] += 1
                acc[t]["best"] = max(acc[t]["best"], 0.30)

    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = a["pos"], a["neg"], a["unc"], a["best"]
        if not (pos or unc) and g_pos and FEATURES["oa_inherit"]:
            # Inherit the whole-joint statement, unless this compartment was separately
            # and explicitly cleared.
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out


def extract(report: str) -> dict:
    """Twelve (score, confidence) pairs, plus the counts the coverage gauge reads."""
    cls = clauses(report)
    out = {}

    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    for tgt in ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"):
        if tgt == "Contusion":
            s, c, npos, nneg = _score_clauses(
                cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt),
                context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None,
                                              DECOY.get(tgt))
        out[tgt] = s
        out[tgt + "__conf"] = c
        out[tgt + "__npos"] = npos
        out[tgt + "__nneg"] = nneg

    # --- synovitis backoff --------------------------------------------------- #
    # Eighty-eight per cent of reports never write the word, and the annotator marks it
    # on nearly half the studies: the label cannot be read off the term alone. What the
    # report does say is whether the joint is wet and irritated - an effusion, a
    # distended bursa, an inflamed fat pad - and that ordering is the only signal
    # available on the silent majority. It enters at low confidence, so it shapes the
    # ranking without asserting a finding.
    if (FEATURES["synovitis_backoff"] and out["Synovitis__npos"] == 0
            and out["Synovitis__nneg"] == 0):
        proxy = sum(1 for c in cls if SYNOVIAL_PROXY.search(c)
                    and _polarity(c) == "positive")
        eff = out["Effusion"]
        prior = 0.30 + 0.30 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out["Synovitis"] = min(0.72, prior)
        out["Synovitis__conf"] = 0.18

    return out

