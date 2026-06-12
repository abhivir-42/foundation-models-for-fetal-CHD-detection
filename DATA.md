# Data

No raw imaging data, and no cached per-subject embeddings derived from it, are
included in this repository.

## The iFIND corpus

The study uses anonymised second-trimester fetal cardiac ultrasound video from
the iFIND research programme at Imperial College London. The corpus is governed
by the iFIND data-sharing agreement and is **not redistributable**. If you
would like to reproduce this work from the raw data, email me at
**as9422@ic.ac.uk** — the corpus is not mine to distribute, but I will gladly
forward your request to the iFIND data custodians at Imperial College London.

What this repository ships instead, so that results remain reproducible without
the raw data:

- the per-fold model **predictions** (`predictions/`), at integer subject-index
  granularity with no identifying information;
- all code to extract embeddings, train heads, and evaluate, should an
  authorised user obtain the corpus.

In practice: anyone can reproduce every reported number, figure, and table
from the committed predictions (`python scripts/reproduce.py`) — no data access
required. Only re-running the pipeline from raw video needs the corpus and the
embeddings derived from it, neither of which is included for the reasons above.

## Cohort and label schema

- **Subjects:** 4,128.
- **Label file:** `subject_level_labels.csv` (expected under `IFIND_DATA`), with
  one binary column per condition plus the subject identifier. The collapse from
  the original fine-grained diagnostic codes to the nine condition groups is
  defined by the iFIND project, not derived here.
- **Conditions (nine):** transposition of the great arteries (TGA),
  atrioventricular septal defect (AVSD), tetralogy of Fallot (TOF),
  hypoplastic left heart syndrome (HLHS), right aortic arch (RAA),
  coarctation of the aorta (CoA), pulmonary atresia, aortic stenosis,
  pulmonary stenosis.

## Prevalence

- **All-nine (any condition vs healthy):** approximately 15.5% of subjects.
- **Auditable-four (TGA, AVSD, TOF, HLHS vs healthy-only):** approximately 10.4%.

These low prevalences are the reason the evaluation reports sensitivity at fixed
specificity, predictive values transported to screening prior, and decision-curve
net benefit, rather than threshold-free discrimination alone.

## Privacy

The committed predictions reference subjects only by an integer index used for
fold assignment. They contain no names, dates, scan identifiers, or any other
potentially identifying field.
