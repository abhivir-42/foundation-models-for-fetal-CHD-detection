"""Full label-disagreement analysis across all 4128 subjects.

For every token-4 string with n>=3 subjects, tabulate 9-col flag distribution.
Quantify per-condition disagreement rate. Propose an "agreement-subset" slice.

Outputs structured data to stdout; the writer turns it into a markdown doc."""

import os as _os_scrub
from pathlib import Path as _Path_scrub
_REPO = _Path_scrub(__file__).resolve().parent.parent.parent
IFIND_DATA     = _os_scrub.environ.get('IFIND_DATA',     str(_REPO/'data'))
EMBEDDINGS_DIR = _os_scrub.environ.get('EMBEDDINGS_DIR', str(_REPO/'embeddings'))
RESULTS_DIR    = _os_scrub.environ.get('RESULTS_DIR',    str(_REPO/'results'))
_CKPT_DIR      = _os_scrub.environ.get('FETALCLIP_DIR',  str(_REPO/'checkpoints'))

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import os

DATA = Path(str(IFIND_DATA))
NON_DOPPLER = Path(str(_Path_scrub(RESULTS_DIR) / "doppler_filter/non_doppler_videos.txt"))
LABELS_CSV = DATA / "subject_level_labels.csv"

CONDITIONS = ['avsd', 'hlhs', 'tga', 'tetralogy', 'raa', 'coa',
              'p_atresia', 'a_stenosis', 'p_stenosis']

# Conservative keyword rules: token4 supports condition X iff one of these
# substrings appears in lowercased token4. Designed to UNDER-count disagreement
# (i.e., a real disagreement might be flagged as agreement, but never vice
# versa) — anything that even loosely names the condition counts as support.
KEYWORDS = {
    'avsd':       ['avsd'],
    'hlhs':       ['hlhs', 'hypoplastic'],
    'tga':        ['tga', 'transposition'],
    'tetralogy':  ['tetralogy', 'tof', 'fallot'],
    'raa':        ['raa'],
    'coa':        ['coa', 'coarctation'],
    # Tricuspid atresia frequently co-presents with pulmonary atresia in the
    # 9-col schema (the cross-tab found token4=DILV / TVD subjects flagged as
    # p_atresia), so accept tricuspid.atresia as supportive too.
    'p_atresia':  ['p.atresia', 'pulmonary.atresia', 'patresia',
                   'tricuspid.atresia'],
    'a_stenosis': ['a.stenosis', 'aortic.stenosis', 'astenosis'],
    'p_stenosis': ['p.stenosis', 'pulmonary.stenosis', 'pstenosis'],
}


def supports(token4_lower: str, cond: str) -> bool:
    if not token4_lower:
        return False
    for kw in KEYWORDS[cond]:
        if kw in token4_lower:
            return True
    return False


def main():
    # Load all-subject labels (CSV has subject + 9 condition cols, no
    # 'unhealthy' column — derive it as OR of the 9 conditions).
    labels = {}
    with open(LABELS_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            sid = int(row['subject'])
            labels[sid] = {c: int(row[c]) for c in CONDITIONS if c in row}
            labels[sid]['unhealthy'] = int(any(labels[sid][c] == 1
                                               for c in CONDITIONS
                                               if c in labels[sid]))

    # Walk filenames, build per-subject token-4 distribution
    subj_tok4 = defaultdict(Counter)
    file_total = 0
    for line in open(NON_DOPPLER):
        path = line.strip()
        if not path:
            continue
        file_total += 1
        stem = os.path.splitext(os.path.basename(path))[0]
        toks = stem.split('_')
        if len(toks) < 4:
            continue
        try:
            sid = int(toks[0])
        except ValueError:
            continue
        if sid not in labels:
            continue
        subj_tok4[sid][toks[3]] += 1

    # Per-subject canonical token-4 = most common
    canon = {}
    for sid, c in subj_tok4.items():
        canon[sid] = c.most_common(1)[0][0] if c else ''

    n_subjects = len(labels)
    n_with_tok4 = sum(1 for s in labels if canon.get(s, '') != '')
    n_unhealthy = sum(1 for s in labels.values() if s['unhealthy'] == 1)
    n_unhealthy_with_tok4 = sum(
        1 for s, v in labels.items()
        if v['unhealthy'] == 1 and canon.get(s, '') != ''
    )

    print(f"## Coverage")
    print(f"- subjects in labels CSV         : {n_subjects}")
    print(f"- subjects with non-Doppler files : {len(subj_tok4)}")
    print(f"- subjects with non-empty token-4 : {n_with_tok4}")
    print(f"- unhealthy subjects               : {n_unhealthy}")
    print(f"- unhealthy with non-empty token-4 : {n_unhealthy_with_tok4} "
          f"({n_unhealthy_with_tok4/n_unhealthy*100:.1f}%)")

    # Distinct token-4 strings, n>=3
    tok4_subjects = defaultdict(list)
    for sid, t in canon.items():
        tok4_subjects[t].append(sid)
    distinct = sorted(tok4_subjects.items(), key=lambda x: -len(x[1]))

    print()
    print(f"## token-4 distribution across {len(canon)} subjects "
          f"(showing values with n>=3)")
    print(f"distinct token-4 values total: {len(tok4_subjects)}")

    # For each token-4 with n>=3, tabulate 9-col flag distribution
    rows = []
    for tok4, sids in distinct:
        if len(sids) < 3:
            continue
        flag_counts = Counter()
        for sid in sids:
            flags = tuple(c for c in CONDITIONS if labels[sid].get(c, 0) == 1)
            flag_counts[flags] += 1
        rows.append((tok4, len(sids), flag_counts))

    print()
    print(f"## token-4 -> 9-col flag set (n>=3 subjects)")
    for tok4, n, flag_counts in rows:
        print(f"\n### token4={tok4!r}  n_subjects={n}")
        for flags, count in flag_counts.most_common():
            label = ','.join(flags) if flags else 'NONE (healthy)'
            print(f"  {label:50s}  n={count}")

    # Per-condition disagreement rate
    print()
    print(f"## Per-condition disagreement rate")
    print()
    print(f"For each 9-col condition X, count subjects with X=1 and bucket by:")
    print(f"  empty       : token-4 is empty")
    print(f"  disagree    : token-4 is non-empty and does NOT mention X "
          f"(per conservative keyword rule)")
    print(f"  agree       : token-4 mentions X")
    print()
    print(f"| condition | n_pos_total | empty | disagree | agree | "
          f"disagreement_rate |")
    print(f"|---|---:|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        n_pos = empty = disagree = agree = 0
        for sid, lab in labels.items():
            if lab.get(cond, 0) != 1:
                continue
            n_pos += 1
            tok = canon.get(sid, '').lower()
            if not tok:
                empty += 1
            elif supports(tok, cond):
                agree += 1
            else:
                disagree += 1
        denom = n_pos if n_pos else 1
        rate = (empty + disagree) / denom
        print(f"| {cond} | {n_pos} | {empty} | {disagree} | {agree} | "
              f"{rate:.2%} |")

    # Propose the agreement subset
    print()
    print(f"## Agreement-subset proposal")
    print()
    print(f"Definition:")
    print(f"  - Healthy subject (unhealthy=0) with empty token-4 -> AGREE")
    print(f"  - Healthy subject with non-empty token-4 -> SKIP "
          f"(token-4 reports a finding the 9-col denies)")
    print(f"  - Unhealthy subject (unhealthy=1) where AT LEAST ONE 9-col "
          f"flag is supported by token-4 -> AGREE")
    print(f"  - Otherwise -> SKIP")
    n_agree = n_skip_healthy_with_tok = n_skip_unhealthy = 0
    per_cond_agree = Counter()
    for sid, lab in labels.items():
        tok = canon.get(sid, '').lower()
        unhealthy = lab.get('unhealthy', 0) == 1
        if not unhealthy:
            if not tok:
                n_agree += 1
            else:
                n_skip_healthy_with_tok += 1
            continue
        # unhealthy
        any_supported = False
        for c in CONDITIONS:
            if lab.get(c, 0) == 1 and supports(tok, c):
                any_supported = True
                per_cond_agree[c] += 1
        if any_supported:
            n_agree += 1
        else:
            n_skip_unhealthy += 1
    print()
    print(f"- agreement subset size: {n_agree} "
          f"(of {n_subjects} subjects, {n_agree/n_subjects:.1%})")
    print(f"- skipped (healthy w/ token-4 finding): {n_skip_healthy_with_tok}")
    print(f"- skipped (unhealthy, no 9-col flag supported by token-4): "
          f"{n_skip_unhealthy}")
    print()
    print(f"### Per-condition n in the agreement subset")
    print()
    print(f"| condition | n_pos (full) | n_pos (agreement subset) | "
          f"shrinkage |")
    print(f"|---|---:|---:|---:|")
    for cond in CONDITIONS:
        n_pos_full = sum(1 for v in labels.values() if v.get(cond, 0) == 1)
        n_pos_sub = per_cond_agree.get(cond, 0)
        shrink = (n_pos_full - n_pos_sub) / n_pos_full if n_pos_full else 0
        print(f"| {cond} | {n_pos_full} | {n_pos_sub} | {shrink:.0%} |")


if __name__ == '__main__':
    main()
