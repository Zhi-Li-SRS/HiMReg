# Lessons

- When user asks to preserve a labeling convention (e.g., `SimpleITK -> Elastix` in figures), do not "fix" naming semantics; isolate stability fixes to optimization and reproducibility paths.
- If a result regresses unexpectedly, verify baseline backend, ROI set, and seed mechanics before tuning model parameters.
- For `case_seed` reproducibility requests, prefer deterministic mappings that preserve historical behavior (`roiN -> base_seed + N - 1`) before introducing hash-based seeds that can shift stochastic optimization outcomes.
- When user marks a preprocessing step as mandatory (e.g., initial spatial rescale), treat it as a hard pipeline invariant and share the exact same implementation between benchmark and main entrypoints.
- When benchmark outputs and reviewer figures live in separate paths (e.g., `roi*/` vs `figures/roi*/`), always refresh figure exports after reruns; stale overlays can look like model regressions.
