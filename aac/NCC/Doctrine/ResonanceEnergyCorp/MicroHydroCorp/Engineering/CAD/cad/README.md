# CAD

## Layout
- `params/params.json` — canonical parameter set (units in mm unless stated)
- `macros/` — FreeCAD macros for part build + export
- `exports/` — generated outputs (STEP/STL/PDF). **Not versioned.**

## Usage (typical)
1. Update `params/params.json`.
2. Run `macros/build_all.FCMacro` inside FreeCAD.
3. Confirm outputs under `exports/`.

Tip: Keep macros deterministic: geometry should be fully defined by `params.json`.

