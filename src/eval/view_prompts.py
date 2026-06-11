"""View-prompt families for FetalCLIP text-encoder view-aware re-weighting (U3).

This file is the prompt corpus used to compute per-clip view-family cosines
against the FetalCLIP text encoder. Each family has >=5 prompts varied in
phrasing but referencing the same fetal cardiac view.

Clinical references:
  - Carvalho JS, Axt-Fliedner R, Chaoui R, Copel JA, Cuneo BF, Goff D, Gordin
    Kopylov L, Hecher K, Lee W, Moon-Grady AJ, Mousa HA, Munoz H, Paladini D,
    Prefumo F, Quarello E, Rychik J, Tutschek B, Wiechec M, Yagel S. ISUOG
    Practice Guidelines (updated): fetal cardiac screening. Ultrasound Obstet
    Gynecol. 2023;61(6):788-803. doi:10.1002/uog.26224
  - AIUM Practice Parameter for the Performance of Fetal Echocardiography (2020).
  - Allan L et al. Guidelines for the performance of the fetal echocardiogram.
    J Am Soc Echocardiogr. 2004;17(7):803-810.
  - International Society of Ultrasound in Obstetrics and Gynecology (ISUOG):
    Cardiac screening examination of the fetus: ISUOG practice guidelines.
    UOG. 2013;41(3):348-359.

Per-condition view mapping (clinically derived; see the pre-registered clinical
mechanism"):
  RAA           -> 3VT  (right aortic arch is a single-view diagnostic;
                   the V-shape of aorta+PA inverts laterally)
  CoA           -> 3VT  (coarctation shows subtle arch narrowing at 3VT)
  TGA           -> 3VT  (parallel outflow tracts visible at 3VT)
  HLHS          -> 4CH  (single dominant chamber across 4-chamber)
  AVSD          -> 4CH  (common atrioventricular valve / inlet)
  TOF           -> multi-plane (VSD + overriding aorta + RV outflow + RVH)
  P_atresia     -> 3VT + outflow (RV outflow obstruction)
  A_stenosis    -> outflow (LV outflow geometry)
  P_stenosis    -> outflow (RV outflow geometry)

The four families below cover this mapping:
  - prompts_4chamber    : 4-chamber and inlet views (HLHS, AVSD)
  - prompts_3vt         : 3-vessel-trachea / 3-vessel views (RAA, CoA, TGA)
  - prompts_outflow     : LVOT, RVOT, outflow tract views
                          (A_stenosis, P_stenosis, P_atresia, partly TOF)
  - prompts_general     : general fetal cardiac sweep anchor; not condition-
                          specific. Used to compute background level so the
                          family scores aren't biased toward "any cardiac-y".

For per-condition derivation under Mode B (auditable-4 focus), the relevant
families are:
  AVSD -> 4CH; HLHS -> 4CH; TGA -> 3VT; TOF -> max(4CH, 3VT, outflow).

For Mode A (max-family), each clip is scored by max across {4CH, 3VT, outflow}
(the general family is excluded from max because it tags non-diagnostic
cardiac sweeps -- we keep it only for sanity diagnostics).
"""
from __future__ import annotations

prompts_4chamber = [
    "Fetal ultrasound image of the four-chamber view of the heart, showing all four cardiac chambers.",
    "Detailed fetal echocardiogram in the apical four-chamber view with both atria and ventricles visible.",
    "Transverse fetal chest ultrasound at the level of the four-chamber view of the heart.",
    "Fetal cardiac four-chamber view showing the atrioventricular valves and ventricular septum.",
    "Ultrasound scan showcasing the fetal heart four chamber view with both atria and ventricles.",
    "Lateral four-chamber view of the fetal heart with the interventricular septum perpendicular to the beam.",
    "Fetal echocardiography apical four-chamber projection of the cardiac inlet.",
]

prompts_3vt = [
    "Fetal ultrasound image of the three-vessel-trachea view of the heart and great arteries.",
    "Three-vessel and trachea view of the fetal upper mediastinum showing pulmonary artery, aorta and superior vena cava with the trachea.",
    "Fetal echocardiographic three-vessel view at the level of the upper thorax.",
    "Transverse fetal ultrasound at the level of the trachea showing the three-vessel arrangement of pulmonary artery, transverse aortic arch and superior vena cava.",
    "Ultrasound scan showcasing the fetal heart three vessel view with pulmonary artery, aorta and superior vena cava.",
    "Fetal upper-thorax cardiac view showing the V-shape confluence of the transverse aortic arch and ductal arch left of the trachea.",
    "Three-vessel-trachea sweep showing aorta, pulmonary trunk and trachea in the same axial plane.",
]

prompts_outflow = [
    "Fetal ultrasound image of the left ventricular outflow tract.",
    "Detailed ultrasound of the fetal right ventricular outflow tract and pulmonary artery origin.",
    "Five-chamber view of the fetal heart showing the aorta arising from the left ventricle.",
    "Fetal echocardiographic outflow-tract view with continuity of the interventricular septum and the aortic root.",
    "Ultrasound of the fetal cardiac outflow tracts demonstrating the crossing of the great arteries.",
    "Long-axis fetal cardiac view of the left ventricular outflow tract and ascending aorta.",
    "Short-axis view at the base of the fetal heart showing the right ventricular outflow tract and pulmonary trunk wrapping the aortic root.",
]

prompts_general = [
    "Fetal ultrasound image of the heart in a general cardiac sweep view.",
    "Fetal echocardiography image showing cardiac anatomy.",
    "Ultrasound of the fetal thorax centered on the heart.",
    "Fetal cardiac ultrasound image (non-specific cardiac plane).",
    "Generic fetal heart ultrasound image with cardiac structures visible.",
]


PROMPT_FAMILIES = {
    "four_chamber": prompts_4chamber,
    "three_vessel": prompts_3vt,
    "outflow":      prompts_outflow,
    "general":      prompts_general,
}

CONDITION_FAMILY = {
    "RAA":        ["three_vessel"],
    "CoA":        ["three_vessel"],
    "TGA":        ["three_vessel"],
    "HLHS":       ["four_chamber"],
    "AVSD":       ["four_chamber"],
    "TOF":        ["four_chamber", "three_vessel", "outflow"],
    "P_atresia":  ["three_vessel", "outflow"],
    "A_stenosis": ["outflow"],
    "P_stenosis": ["outflow"],
}

DIAGNOSTIC_FAMILIES = ["four_chamber", "three_vessel", "outflow"]

AUDIT4_RELEVANT_FAMILIES = ["four_chamber", "three_vessel", "outflow"]


def all_diagnostic_prompts():
    """Flat list of all diagnostic (non-general) prompts; useful for sanity."""
    out = []
    for fam in DIAGNOSTIC_FAMILIES:
        out.extend(PROMPT_FAMILIES[fam])
    return out


if __name__ == "__main__":
    for k, v in PROMPT_FAMILIES.items():
        print(f"[{k}] {len(v)} prompts")
        for p in v:
            print(f"  - {p}")
    print()
    print(f"Diagnostic families (used for max-family bias): {DIAGNOSTIC_FAMILIES}")
    print(f"Auditable-4 relevant families: {AUDIT4_RELEVANT_FAMILIES}")
    print(f"Per-condition mapping ({len(CONDITION_FAMILY)} conditions):")
    for c, fs in CONDITION_FAMILY.items():
        print(f"  {c:12s} -> {fs}")
