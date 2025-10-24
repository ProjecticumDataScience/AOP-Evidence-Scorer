"""
The script runs a "classic" NER for two types of entities — CHEMICAL and SPECIES — on texts from articles, listed in Excel (Paper column).
It loads pre-trained biomedical models from spaCy,
extracts entities from the column with agent responses,
compiles the results (including coordinates and the context sentence),
prints summary statistics, and
saves the results in a CSV file.
"""



# agent_gene_triplets.py
# Agent → action → target (gene) extractor with fixes for:
# "Hepatocyte VDR activation upregulates ANGPTL8"
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
import os
import re
import sys
import unicodedata

# =========================
# Models (CHEM only; genes —from  extern dictionary)
# =========================
try:
    import spacy
except Exception as e:
    print("spaCy required. Install:\n  pip install spacy scispacy")
    sys.exit(1)

try:
    chem_nlp = spacy.load("en_ner_bc5cdr_md")
except Exception as e:
    print(
        "Model 'en_ner_bc5cdr_md' not installed.\n"
        "Install:\n"
        "  pip install scispacy\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz\n"
        f"Original error: {e}"
    )
    sys.exit(1)

# =========================
# Config / paths
# =========================
#DICT_PATH = Path(os.getenv("GENE_DICT_TSV", r"R:\gene_info\gene_dict.tsv")).resolve()
DICT_PATH = Path(os.getenv("GENE_DICT_TSV", "data_input/gene_info/gene_dict.tsv")).resolve()
# =========================
# Normalization
# =========================
_ZERO_WIDTH = r"[\u200B-\u200F\uFEFF\u2060\u00AD]"
_DASHES     = r"[‐-‒–—―﹘﹣－]"
_GREEK_MAP  = str.maketrans({
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta","η":"eta","θ":"theta",
    "ι":"iota","κ":"kappa","λ":"lambda","μ":"mu","ν":"nu","ξ":"xi","ο":"omicron","π":"pi",
    "ρ":"rho","σ":"sigma","τ":"tau","υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega",
    "Α":"alpha","Β":"beta","Γ":"gamma","Δ":"delta","Ε":"epsilon","Ζ":"zeta","Η":"eta","Θ":"theta",
    "Ι":"iota","Κ":"kappa","Λ":"lambda","Μ":"mu","Ν":"nu","Ξ":"xi","Ο":"omicron","Π":"pi",
    "Ρ":"rho","Σ":"sigma","Τ":"tau","Υ":"upsilon","Φ":"phi","Χ":"chi","Ψ":"psi","Ω":"omega",
})

def normalise_biomed(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(_ZERO_WIDTH, "", s)
    s = re.sub(_DASHES, "-", s)
    s = s.translate(_GREEK_MAP)

    s = re.sub(r"\b([A-Za-z0-9]+)-(induced|dependent|mediated|triggered|treated)\b",
               r"\1 \2", s, flags=re.I)
    s = re.sub(r"\bcoa\b", "CoA", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

# ===== норм. для словарного поиска =====
#This normalization is used when loading the alias dictionary (load_gene_aliases), when searching for mentions of genes in the text (extract_genes_with_spans)

GREEK_LATIN = str.maketrans({"α":"A","β":"B","γ":"G","δ":"D","κ":"K","μ":"M","ν":"N",
                             "Α":"A","Β":"B","Γ":"G","Δ":"D","Κ":"K","Μ":"M","Ν":"N"})
def norm_lex(s: str) -> str:
    s = (s or "").translate(GREEK_LATIN).upper().strip()
    return s.replace("XBP-1S","XBP1").replace("XBP-1","XBP1").replace("IRE1Α","IRE1A")

# =========================
# loads a dictionary of genes (TSV) and builds a fast mapping from it alias → official name
#to recognize genes in the text by any of their synonyms and reduce them to one “official” symbol

def load_gene_aliases(dict_path: Path) -> tuple[Dict[str,str], Set[str]]:
    if not dict_path.exists():
        print(f"Gene dictionary not found: {dict_path}")
        sys.exit(1)
    alias2off: Dict[str,str] = {}
    with dict_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        five_cols = (header[:5] == ["tax","gid","official","short_aliases","long_names"])
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if five_cols:
                tax, gid, official, shorts, _ = parts
            else:
                tax, gid, shorts, _ = parts[0], parts[1], parts[2], "-"
                official = shorts.split("|")[0]
            if tax != "9606":      # только human, как у вас
                continue
            official = norm_lex(official)
            for a in shorts.split("|"):
                a = norm_lex(a)
                if a:
                    alias2off.setdefault(a, official)
    return alias2off, set(alias2off)

ALIAS2OFF, ALIASES = load_gene_aliases(DICT_PATH)

# =========================
# Lemma/verb direction (без spaCy)
# =========================
INCREASE_LEMMAS = {
    "increase","upregulate","induce","activate","enhance","elevate","augment",
    "stimulate","boost","raise","amplify","potentiate","promote","trigger","drive",
    "upregulation","activation",
}
DECREASE_LEMMAS = {
    "decrease","downregulate","repress","suppress","reduce","inhibit","attenuate",
    "alleviate","mitigate","diminish","lower","deplete","block","prevent","impair",
    "downregulation","inhibition",
}

_LEMMA_NORM = {
    "increases":"increase","increased":"increase","increasing":"increase",
    "decreases":"decrease","decreased":"decrease","decreasing":"decrease",
    "upregulated":"upregulate","upregulates":"upregulate",
    "downregulated":"downregulate","downregulates":"downregulate",
    "activates":"activate","activated":"activate","activating":"activate",
    "promotes":"promote","promoted":"promote",
    "inhibits":"inhibit","inhibited":"inhibit","inhibiting":"inhibit",
    "suppresses":"suppress","represses":"repress",
    "reduces":"reduce","reduced":"reduce",
    "lowers":"lower","raises":"raise","elevates":"elevate",
}

def _basic_lemmas(text)  :
    toks = [w.lower() for w in re.findall(r"[A-Za-z0-9\-]+", text)]
    return [_LEMMA_NORM.get(t, t) for t in toks]

def lemma_ngrams(text)  :
    toks = _basic_lemmas(text)
    return set(toks) | {f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)}

def direction_from_lemmas(text) :
    grams = lemma_ngrams(text)
    if grams & DECREASE_LEMMAS: return "decrease"
    if grams & INCREASE_LEMMAS: return "increase"
    return None

# =========================
# Regex for agent
# =========================
AGENT_CORE   = r"[A-Za-z0-9µμα-ωΑ-Ω][A-Za-z0-9µμα-ωΑ-Ω\-–—\(\)\[\]/\+\.]*"
AGENT_SEQ    = rf"{AGENT_CORE}(?:\s+{AGENT_CORE}){{0,5}}"
AGENT_PHRASE = rf"({AGENT_SEQ})"
HYPHEN_INDUCED = rf"({AGENT_SEQ})-induced"
DIET_PHRASE  = rf"((?:high|low|chronic|acute|western|cafeteria|fat|fructose|sucrose|cholesterol|methionine|choline|deficient|rich|enriched|control)\s+{AGENT_CORE}(?:\s+{AGENT_CORE}){{0,3}}\s+(?:diet|intake|feeding))"

# combo: второй «агент» не должен начинаться с герундия
_COMBO = rf"\b{AGENT_PHRASE}\s+(?:and|plus|with|&|/)\s+(?!promoting\b|increasing\b|decreasing\b|reducing\b){AGENT_PHRASE}\b"

_AGENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf"\b(?:induced|triggered|mediated|worsened|exacerbated|aggravated)\s+by\s+{AGENT_PHRASE}\b", re.I), "induced_by"),
    (re.compile(rf"\bby\s+{AGENT_PHRASE}\b", re.I), "by"),
    (re.compile(rf"\b(?:co-?treated|treated|pretreatment)\s+with\s+{AGENT_PHRASE}\b", re.I), "treated_with"),
    (re.compile(rf"\binjected[^.]{{0,120}}?\bwith\s+{AGENT_PHRASE}\b", re.I), "injected_with"),
    (re.compile(rf"\b(?:administration|coadministration|bolus|dose)\s+of\s+{AGENT_PHRASE}\b", re.I), "of_administration"),
    (re.compile(rf"\bexposure\s+(?:to|of)\s+{AGENT_PHRASE}\b", re.I), "exposure_to"),
    (re.compile(rf"\b{AGENT_PHRASE}\s+exposure\b", re.I), "noun_exposure"),
    (re.compile(HYPHEN_INDUCED, re.I), "hyphen_induced"),
    # ⬇⬇ добавили upregulates / activates и downregulates / inhibits
    (re.compile(rf"\b{AGENT_PHRASE}\s+(?:induces?|provokes?|causes?|triggers?|worsens?|exacerbates?|promotes?|drives?|elevates?|raises?|increases?|upregulates?|activates?|leads?\s+to)\b", re.I), "verb_positive"),
    (re.compile(rf"\b{AGENT_PHRASE}\s+(?:reduces?|lowers?|attenuates?|alleviates?|mitigates?|protects?\s+against|prevents?|reverses?|ameliorates?|suppresses?|downregulates?|inhibits?)\b", re.I), "verb_protective"),
    (re.compile(rf"\b{DIET_PHRASE}\b", re.I), "diet_phrase"),
    (re.compile(_COMBO, re.I), "combo_first"),
    (re.compile(rf"\b{AGENT_PHRASE}\s+(?:induced|dependent|mediated|triggered|treated)\b", re.I), "hyphen"),
    # роли/функции
    (re.compile(rf"\b{AGENT_PHRASE}\s+plays?\s+(?:an?\s+)?(?:critical|important|key)?\s*roles?\b", re.I), "plays_role"),
    (re.compile(rf"\b(?:impairment|activation|inhibition|loss|reduction|increase)\s+of\s+{AGENT_PHRASE}\s+(?:activity|function)\b", re.I), "of_activity")
    ]


#  verb_positive, verb_protective, treated_with, injected_with, of_administration, induced_by, plays_role, of_activity - that is, the formulations that most reliably indicate the agent, see also Agent-patterns

_STRONG = {"verb_positive","verb_protective","treated_with","injected_with","of_administration","induced_by","plays_role","of_activity"}

# cleaning up agent candidates from lists
_CHEM_STOP = {"glucose","serum glucose","triglycerides","cholesterol","fatty acid","fatty acids","carbohydrate","lipid","lipids","metabolism"}
_BY_CUT = {"pretreatment","treatment","injection","administration","exposure","pretreated","treated","patients","mice","rats","cells","cell","in","of","on","at","with","to","by","for","from","under","during"}
_BAD_HEADS = {"the","a","an","of","mixture","solution"}
_BAD_NOUNS = {"decrease","increase","attenuation","expression","expressions","level","levels","gene","genes","change","changes","activity","function"}

#  This is a "cutter" for   abbreviations
#"peroxisome proliferator-activated receptor alpha (PPARα)" → "peroxisome proliferator-activat

def _trim_after_stop(phrase: str) -> str:
    toks, out = phrase.split(), []
    for t in toks:
        if t.lower() in _BY_CUT:
            break
        out.append(t)
    return " ".join(out) if out else phrase.strip()

def _strip_paren_acronym(text: str) -> str:
    m = re.match(r"^(?P<long>[^()]{3,})\s*\((?:[^)]+)\)\s*$", text.strip())
    return m.group("long").strip() if m else text.strip()

#    a tail trimmer for the agent candidate: it removes abstract nouns like activation/activity/function/expression/pathway/signaling from the end.
#  "β2-adrenergic receptor activation by formoterol" → "β2-adrenergic receptor"

def _drop_trailing_noun(text: str) -> str:

    return re.sub(r"\s+(?:activation|activity|function|expression|pathway|signaling)$", "", text.strip(), flags=re.I)

# a mini-sanitizer for the agent candidate extracted by regex, and it is used specifically for cases of the pattern “… by X
 #by the solution of sodium acetate with water  ->  sodium acetate 

def _refine_agent_with_chem(segment: str) -> Optional[str]:
    seg = segment.strip()
    cand = _trim_after_stop(seg)
    parts = cand.split()
    while parts and parts[0].lower() in _BAD_HEADS:
        parts.pop(0)
    if not parts:
        return None
    if parts[0].lower() in _BAD_NOUNS:
        return None
    return " ".join(parts)

# =========================
# Gene finder (by your dictionary)
# =========================
WORD_RE = re.compile(r"[A-Za-z0-9\-α-ωΑ-Ω]+")
AMBIGUOUS = {"MICE","AS","OF","IN","ON","AND","OR","TO","BY","THE","EP","ER"}

#  function runs through the words in a sentence and, relying on a dictionary of aliases, marks those that are genes, returning the official symbol and span.

def extract_genes_with_spans(text: str) -> List[Dict]:
    seen: Dict[str, Dict] = {}
    for m in WORD_RE.finditer(text):
        word = m.group(0)
        if word.islower():
            continue
        cands = {norm_lex(word), norm_lex(word.replace("-",""))}
        for t in list(cands):
            if not t or t.isdigit() or t in AMBIGUOUS:
                continue
            if t in ALIASES:
                off = ALIAS2OFF[t]
                seen.setdefault(off, {"alias": t, "start": m.start(), "end": m.end()})
                break
    return [{"official": off, **v} for off, v in sorted(seen.items())]

# assign direction per gene (rule-based + window)
_UP = r"(?:increase|increased|upregulate|upregulated|upregulation|elevate|elevated|induce|induced|promote|promoted|enhance|enhanced|activate|activated|activation|upregulates?)"
_DOWN = r"(?:decrease|decreased|downregulate|downregulated|downregulation|reduce|reduced|suppress|suppressed|inhibit|inhibited|attenuate|attenuated|alleviate|alleviated|repress|repressed|downregulates?)"
_LIST = r"([A-Za-z0-9\-αΑ]+(?:\s*,\s*[A-Za-z0-9\-αΑ]+)*(?:\s*,?\s*(?:and|и)\s*[A-Za-z0-9\-αΑ]+)?)"

def _split_list(s: str) -> List[str]:
    parts = re.split(r"\s*,\s*|\s*(?:and|и)\s*", s)
    return [p for p in (x.strip() for x in parts) if p]

def assign_gene_directions(text: str, mentions: List[Dict]) -> List[Dict]:
    up_hits, down_hits = set(), set()
    for rx, bucket in (
        (rf"{_UP}\s+(?:the\s+)?(?:gene\s+expression\s+of\s+)?{_LIST}", up_hits),
        (rf"{_DOWN}\s+(?:the\s+)?(?:gene\s+expression\s+of\s+)?{_LIST}", down_hits),
        (rf"{_UP}[^.]*?\bincluding\s+{_LIST}", up_hits),
        (rf"{_DOWN}[^.]*?\bincluding\s+{_LIST}", down_hits),
    ):
        for m in re.finditer(rx, text, flags=re.I):
            for g in _split_list(m.group(1)):
                t = norm_lex(g)
                if t in ALIASES:
                    bucket.add(ALIAS2OFF[t])

    # "including LIST ... which were downregulated ..."
    for m in re.finditer(rf"\bincluding\s+{_LIST}[^.]*?\bwhich\s+were\s+({_DOWN})", text, flags=re.I):
        for g in _split_list(m.group(1)):
            t = norm_lex(g)
            if t in ALIASES:
                down_hits.add(ALIAS2OFF[t])

    def window_dir(lo: int, hi: int) -> Optional[str]:
        L = max(0, lo - 80); R = min(len(text), hi + 80)
        return direction_from_lemmas(text[L:R])

    out = []
    for m in mentions:
        off = m["official"]
        if off in down_hits: d = "decrease"
        elif off in up_hits: d = "increase"
        else: d = window_dir(m["start"], m["end"])
        out.append({**m, "direction": d})
    return out

# =========================
# Agent extractor + Triplets
# =========================
_GERUNDS = {"promoting","increasing","decreasing","reducing","attenuating","alleviating","mitigating"}

def extract_agent(sent_text: str, *, verbose: bool = True) -> Dict[str, Optional[str]]:
    s = normalise_biomed(sent_text)
    if verbose:
        print("="*80)
        print("SENT:", s)

    def ok(cand: str) -> bool:
        if not cand:
            return False

        cl = cand.lower()
        if cl in _CHEM_STOP:
            print(f"[ok] drop (stoplist): {cand}")
            return False

        # Проверяем «всё капсом» и сверяемся со словарём генов
        if re.fullmatch(r"[A-Z0-9\-]{2,}", cand):
            is_gene = norm_lex(cand) in ALIASES
            print(f"[ok] ALLCAPS={cand!r} -> is_gene={is_gene}")
            if is_gene:
                return False

        print(f"[ok] keep: {cand}")
        return True


    # 1) regex phase
    matches: List[Tuple[str, str, str, Tuple[int,int]]] = []
    for rx, label in _AGENT_PATTERNS:
        for m in rx.finditer(s):
            agent = _drop_trailing_noun(_strip_paren_acronym(m.group(1).strip()))
            if agent.split()[0].lower() in _GERUNDS:
                continue
            span = (m.start(1), m.end(1))
            if label == "by":
                t = agent.lower()
                if t.endswith("ing") or " pathway" in t or " signaling" in t:
                    if verbose: print(f"[regex] skip BY-process: {agent!r}")
                    continue
            if ok(agent):
                matches.append((label, agent, m.group(0), span))
                if verbose:
                    print(f"[regex] + {label:16s} agent={agent!r} span={span} ctx={m.group(0)!r}")

    # 2) no regex → fallback (GENE-list → CHEM)
    if not matches:
        mentions = extract_genes_with_spans(s)
        if mentions:
            best = max(mentions, key=lambda m: len(m["official"]))
            if verbose:
                print(f"[fallback] GENE pick: {best['official']!r} (alias {best['alias']})")
            return {"agent": best["official"], "by": None, "agent source": "gene_dict", "span": (best["start"], best["end"])}

        doc = chem_nlp(s)
        chems = [e.text for e in doc.ents if e.label_ == "CHEMICAL" and ok(e.text)]
        if chems:
            pick = max(chems, key=len)
            if verbose:
                print(f"[fallback] CHEM pick: {pick!r}")
            return {"agent": pick, "by": None, "agent source": "chem", "span": None}
        if verbose:
            print("[result] no agent")
        return {"agent": None, "by": None, "agent source": None, "span": None}

    # 3) rank regex matches
    def base_rank(lbl: str) -> int:
        if lbl in _STRONG: return 0
        if lbl == "by":    return 2
        return 1

    ranked: List[Tuple[int,str,str,str,Tuple[int,int]]] = []
    for (lbl, agent, ctx, span) in matches:
        r = base_rank(lbl)
        ranked.append((r, lbl, agent, ctx, span))
        if verbose:
            print(f"[rank] lbl={lbl:16s} agent={agent!r} => rank={r}")
    if any(lbl in _STRONG for _, lbl, *_ in ranked):
        ranked = [t for t in ranked if not (t[1] == "by" and t[0] >= 2)]
    ranked.sort(key=lambda x: (x[0], -len(x[2])))

    _, lbl, agent, ctx, span = ranked[0]
    if verbose:
        print(f"[pick] top= {lbl=} {agent=!r}  ctx={ctx!r}")

    if lbl == "by":
        refined = _refine_agent_with_chem(agent)
        if refined:
            agent = refined
            if verbose: print(f"[refine] BY cleanup -> {agent!r}")

    return {"agent": agent, "by": lbl, "agent source": "regex", "span": span}

def direction_from_label_or_context(lbl: Optional[str], text: str, span: Optional[Tuple[int,int]]) -> Optional[str]:
    if lbl in {"verb_positive","treated_with","injected_with","of_administration","induced_by","plays_role"}:
        return "increase"
    if lbl in {"verb_protective"}:
        return "decrease"
    if span:
        L = max(0, span[0]-80); R = min(len(text), span[1]+80)
        return direction_from_lemmas(text[L:R])
    return direction_from_lemmas(text)

def build_triplets(sentence: str, agent_block: Dict, *, verbose: bool = True) -> Dict:
    s = normalise_biomed(sentence)
    agent = agent_block.get("agent")
    lbl   = agent_block.get("by")
    span  = agent_block.get("span")

    mentions = extract_genes_with_spans(s)
    mentions = assign_gene_directions(s, mentions)

    edge_dir = direction_from_label_or_context(lbl, s, span)
    triplets = []
    for g in mentions:
        d = g.get("direction") or edge_dir
        if not d:
            continue
        triplets.append({"agent": agent, "verb_dir": d, "target": g["official"],
                         "target_alias": g["alias"], "target_span": (g["start"], g["end"])})
    if verbose:
        print("GENE TARGETS:")
        for t in triplets:
            print(f"  {t['agent']}  --{t['verb_dir']}-->  {t['target']}  (as {t['target_alias']}) {t['target_span']}")
    return {"agent": agent, "triplets": triplets}

# =========================


EXAMPLES = [
    "TCDD-induced thioesterases create a futile acyl‑CoA activation–hydrolysis cycle that disrupts hepatic β‑oxidation.",
    "Elevated intracellular S‑adenosylhomocysteine drives hepatocyte triglyceride accumulation by boosting lipogenesis and suppressing lipolysis.",
    "Dietary α‑lactalbumin shifts hepatic metabolism toward lipogenesis and away from fatty acid oxidation, inducing steatosis.",
    "Caveolin‑1 mitigates acetaminophen‑aggravated lipid accumulation in alcoholic fatty liver by activating Pink1/Parkin‑dependent mitophagy.",
    "Acetoacetic acid triggers oxidative stress in hepatocytes, impairing VLDL assembly and promoting triglyceride accumulation.",
    "Metformin reduces lipid accumulation in hepatocytes induced by oleic acid and CB1 receptor agonists, implicating endocannabinoid signaling.",
    "Amiodarone provokes ER stress that drives lipid droplet protein expression and triglyceride buildup; blocking ER stress prevents steatosis.",
    "Dietary trans fats and anabolic steroids interact to worsen hepatic lipotoxicity and NAFLD by enhancing triglyceride accumulation and oxidative stress.",
    "A high arachidonic acid to DHA ratio induces mitochondrial dysfunction and shifts lipid metabolism toward triglyceride accumulation.",
    "Chronic soft drink or aspartame intake causes oxidative stress and liver injury via adipocytokine dysregulation and altered lipid profile.",
    "Curcumin suppresses benzo[a]pyrene‑induced lipid accumulation and ROS in hepatocytes by downregulating AhR/CYP1A1/CYP1B1 signaling.",
    "Benzbromarone aggravates hepatic steatosis under high free fatty acids by altering lipid metabolism, inflammation, and apoptosis.",
    "Bisphenol‑A exposure exacerbates steatosis by inducing ER stress, activating fibrogenic pathways, and disturbing lipid metabolism.",
    "Low‑dose bisphenol A induces hepatic cholesterol synthesis and steatosis by hypomethylating and upregulating SREBP‑2.",
    "Silybin alleviates BPA‑induced oxidative stress, aberrant proliferation, and steroid hormone oxidation in HepG2 cells.",
    "Acetic acid activates AMPK in hepatocytes, promoting fatty acid oxidation and suppressing lipogenesis to reduce triglycerides.",
    "Chronic cadmium exposure suppresses SIRT1 signaling, impairing mitochondrial fatty acid oxidation and promoting NAFLD.",
    "Low palmitate boosts mitochondrial metabolism via a CDK1–SIRT3–CPT2 cascade, enhancing fatty acid oxidation and stress resistance.",
    "Oral Nigella sativa oil mitigates cisplatin‑induced hepatotoxicity by enhancing antioxidant defenses and metabolic function.",
    "Clozapine worsens NAFLD and metabolic injury in obese mice by increasing hepatic lipid accumulation, oxidative stress, and insulin resistance.",
    "In CLA‑induced steatosis, PPARα is co‑opted to drive both lipogenesis and fatty acid oxidation, reshaping lipid homeostasis.",
    "Dexamethasone exacerbates palmitate‑induced lipotoxicity in HepG2 cells by upregulating fatty acid transport and accumulating TAG/DAG/ceramides.",
    "lncRNA ENST00000608794 promotes dexamethasone‑induced steatosis by sponging miR‑15b‑5p and derepressing PDK4.",
    "Chronic glucocorticoids activate an ANGPTL4→ceramide→PKCζ axis to stimulate de novo lipogenesis and hypertriglyceridemia.",
    "Doxorubicin activates PPARα‑dependent adipose lipolysis, elevating circulating fatty acids that drive hepatic steatosis and insulin resistance.",
    "DGAT1 inhibitor DS‑7250 paradoxically worsens steatosis by upregulating de novo lipogenesis independent of DGAT1 blockade.",
    "Efavirenz activates PXR, inducing hypercholesterolemia and hepatic steatosis via increased lipid uptake and cholesterol synthesis.",
    "Erratum correcting the efavirenz–PXR study on hypercholesterolemia and hepatic steatosis.",
    "Peroxisomal oxidation of erucic acid raises hepatic malonyl‑CoA, suppressing mitochondrial fatty acid oxidation.",
    "Estradiol promotes hepatic lipid deposition in tilapia by upregulating triglyceride synthesis and VLDL assembly while reducing receptor‑mediated clearance.",
    "Apigenin protects against alcohol‑induced liver injury by suppressing CYP2E1‑mediated oxidative stress and modulating PPARα/lipogenic genes.",
    "An aldose reductase inhibitor alleviates ethanol‑induced steatosis by repressing saturated fatty acid biosynthesis.",
    "Vitamin D deficiency unexpectedly attenuates acute alcohol‑induced hepatic lipid accumulation by suppressing lipogenesis and enhancing β/ω‑oxidation.",
    "Impaired TFEB‑mediated lysosome biogenesis and autophagy promotes chronic ethanol‑induced steatosis and liver injury.",
    "CYP2A5 deletion aggravates alcoholic fatty liver via dysregulation of the PPARα–FGF21 axis.",
    "Free fatty acids activate mTORC1 and ER stress to suppress LAMP2 and autophagy, contributing to alcohol‑related liver injury.",
    "DEPTOR loss in ALD increases lipogenesis; DEPTOR overexpression suppresses SREBP‑1, restores FAO, and ameliorates injury.",
    "Ginseng stem/leaf ginsenosides protect hepatocytes from ethanol‑induced lipid accumulation via antioxidant and PPARα‑mediated mechanisms.",
    "NIK links inflammation to alcoholic steatosis by recruiting MEK/ERK to inhibit PPARα and fatty acid oxidation.",
    "Ligustrazine requires Nrf2 activation to suppress SREBP‑1c, induce PPARα, and reduce alcohol‑induced steatosis.",
    "α‑Linolenic acid‑rich flaxseed oil protects against ethanol‑induced steatosis by restoring adipose–liver lipid homeostasis and AMPK signaling.",
    "In ADH‑deficient deer mice, dysregulated AMPK and ER stress synergize to promote ethanol‑induced steatosis and injury.",
    "Hepatic DEPDC5 deficiency hyperactivates mTORC1, suppresses PPARα, and worsens ethanol‑induced steatosis and inflammation.",
    "Ethanol inhibits hepatocyte lipophagy by inactivating Rab7, leading to lipid droplet accumulation.",
    "CYP2E1‑mediated oxidative stress impairs Akt activation, contributing to chronic ethanol‑induced fatty liver.",
    "Liver‑specific PTEN knockout increases lipogenesis yet protects from ethanol‑induced damage by boosting antioxidant capacity.",
    "Hepatic PPARγ drives alcohol‑induced steatosis and inflammation; its knockdown reduces lipogenesis and injury.",
    "TLR4 mutation confers resistance to acute alcohol‑induced SREBP‑1 activation and hepatic triglyceride accumulation.",
    "DPP‑4 inhibition lowers hepatic triglycerides in insulin‑resistant female rats by suppressing uric acid via ADA/XO.",
    "High‑throughput PTM profiling enables molecular classification of fatty liver across etiologies.",
    "Fluoxetine induces hepatic lipid accumulation via PTGS1 upregulation and elevated 15‑deoxy‑Δ12,14‑PGJ2.",
    "β2‑adrenergic receptor activation by formoterol enhances de novo lipogenesis, causes incomplete β‑oxidation, and reduces TAG secretion.",
    "Citrulline supplementation partially prevents fructose‑induced hypertriglyceridemia and hepatic fat accumulation.",
    "Glyphosate exposure disrupts lipid metabolism with oxidative stress and inflammation, causing hepatic fat accumulation in carp.",
    "Omega‑3 PUFAs alleviate hyperhomocysteinemia‑induced steatosis by suppressing hepatic ceramide synthesis.",
    "Linagliptin improves steatosis from dual IR/IGF‑1R inhibition via mechanisms likely involving PLIN2/NNMT rather than classic insulin pathways.",
    "Early microcystin‑LR exposure activates NLRP3 inflammasome and, with later high‑fat diet, promotes NAFLD and insulin resistance.",
    "A miR‑378↔Nrf1 negative feedback loop inhibits fatty acid oxidation and drives high‑fat diet‑induced hepatosteatosis.",
    "MEHP promotes lipid accumulation via JAK2/STAT5 inhibition and oxidative stress in hepatocytes.",
    "Multi‑walled carbon nanotubes disturb lipophagy and cause ER/lysosomal stress, leading to lipid accumulation in HepG2 cells.",
    "NAMPT inhibition lowers NAD+, suppresses SIRT1/AMPKα, increases SREBP1, and aggravates high‑fat diet‑induced steatosis.",
    "Sodium acetate counters nicotine‑induced hepatic lipid excess by inhibiting xanthine oxidase and oxidative stress.",
    "Intestinal FXR via the FXR–FGF15/19 axis suppresses hepatic FAO by inhibiting CREB–PGC1α signaling.",
    "Bee‑Bee Tree oil activates AMPK and inhibits JNK to reduce palmitate‑induced lipid accumulation and apoptosis in hepatocytes.",
    "Palmitate represses FoxO1, reducing ATGL/CGI‑58 and increasing PPARγ/G0S2, thereby impairing lipolysis and increasing fat.",
    "Fermented soymilk reduces hepatocellular steatosis by inhibiting SREBP‑1 and activating NRF‑2.",
    "PFBS increases triglyceride accumulation via PPARγ‑linked lipogenesis, fatty acid uptake, ROS, and ER stress in HepG2 cells.",
    "Details unavailable; abstract not accessible—unable to extract a reliable one‑sentence conclusion.",
    "Chronic phenanthrene exposure causes gut dysbiosis and disrupts hepatic lipid metabolism, increasing liver fat.",
    "Genistein activates ERβ to suppress lipogenesis, enhance β‑oxidation, inhibit Akt/mTOR, and lower hepatic triglycerides.",
    "PM2.5 induces hepatic steatosis mediated by FXR downregulation; FXR loss abrogates PM2.5‑driven lipid accumulation.",
    "PCB126 disrupts gluconeogenesis and peroxisomal FAO via PPARα pathway suppression before overt steatosis.",
    "After Roux‑en‑Y bypass, hepatic mTOR→AKT2→Insig2 signaling suppresses de novo lipogenesis and improves steatosis.",
    "FABP4 inhibitor I‑9 co‑treatment mitigates rosiglitazone‑induced fatty liver without blunting antidiabetic efficacy.",
    "SIRT1 disruption in human fetal hepatocytes elevates intracellular glucose and lipids via increased lipogenesis/gluconeogenesis and reduced AKT/FOXO1 signaling.",
    "SCD1 inhibition activates AMPK and lipophagy to reduce hepatic lipid accumulation.",
    "Valproate induces steatosis by upregulating CD36 and DGAT2 to increase FA uptake and triglyceride synthesis.",
    "Omega‑3 PUFAs prevent LXR‑driven hepatic lipid accumulation via FFA4 (GPR120)–CaMKK–AMPK signaling.",
    "Bilberry fruit extracts limit hepatic fat and injury by curbing lipid accumulation, boosting antioxidant defenses, and promoting lipophagy.",
    "H3K9 demethylase JMJD2B upregulates PPARγ2 epigenetically, driving hepatic triglyceride accumulation.",
    "Tebuconazole causes lipid build‑up by increasing FA uptake, disrupting mitochondria, raising ROS, and lowering MTTP in hepatocytes.",
    "Bicyclol attenuates tetracycline‑induced fatty liver by inhibiting ER stress and apoptosis.",
    "A non‑β‑oxidizable FA analog inhibits mitochondrial FAO, raises NAD+/NADH, alters kynurenine pathway, and causes hepatic TAG accumulation.",
    "Magma seawater reduces liver lipid accumulation by downregulating lipogenesis/cholesterologenesis and upregulating CPT1 and antioxidants.",
    "Tributyltin lowers adiponectin and hepatic/muscle AKT activity, provoking fat gain, dyslipidemia, and insulin resistance.",
    "Tunicamycin‑induced ER stress increases hepatic triglycerides and lowers glycogen, altering lipogenic/lipoprotein gene expression.",
    "Exendin‑4 alleviates ER stress‑induced hepatic lipid accumulation by enhancing lipolysis and VLDL assembly under SIRT1/AMPK control.",
    "High uric acid drives hepatic fat via oxidative stress, JNK activation, and AP‑1‑mediated lipogenic gene induction.",
    "Valproic acid provokes steatosis through a CYP2E1→ROS→CD36/DGAT2 axis; antioxidants blunt these effects.",


]

for s in EXAMPLES:
    blk = extract_agent(s, verbose=True)
    out = build_triplets(s, blk, verbose=True)
    print("\nFINAL:", out, "\n")


