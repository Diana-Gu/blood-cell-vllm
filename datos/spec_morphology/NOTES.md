# Phase 0 Notes

## Morphology KB normalization defaults
- The spreadsheet `general_features` allowed vocab misses values used in prototypes; `spec/vocab_normalization.yaml` captures the fixes used when building `spec/morphology_kb.yaml`.
- Nucleus shape normalization:
  - "segmented/band" -> "band" (band neutrophils use `band`; segmented neutrophils use `segmented`).
  - "round" -> "regular".
  - "cup like" -> "cup-like".
- Nucleus segments normalization:
  - "3 to 5" -> "3-5".
  - "> 5" -> ">5".
  - "variable" -> "undefined".
- Nuclear chromatin: "low condensed" -> "homogeneous/open/low condensed".
- Cytoplasm: "wde" -> "wide".
- Inclusions: "Vauoles" -> "vacuoles"; "Splinters"/"Auer rods" lowercased.
- Granulation: added allowed value "absent" to cover abnormal prototypes.
- Values are lowercased in the KB; cell types are title-cased.

## Cell type naming
- "NEUTROPHIL" -> "Neutrophil Segmented".
- "NEUTROPHIL - BAND" -> "Neutrophil Band".
- Other cell types are title-cased unless explicitly mapped.

## Regeneration
- The KB is generated via `python -m pbcell.captions.kb_build --xlsx spec/pathologist_morphology.xlsx --kb_out spec/morphology_kb.yaml --norm spec/vocab_normalization.yaml`.
