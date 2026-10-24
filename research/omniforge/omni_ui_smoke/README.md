# Omni UI mode: end-to-end smoke test on ROPY-PC (2026-10-24)

- Code: `e2ed11f` ("Omni người (thử nghiệm, GPU)" mode). Called through the running web UI API with `scripts/omni_smoke.py` (POST `/api/generate`, mode omni, isolation contour, depth body).
- Input: a real photo provided by the owner (1024x725 JPG). The photo and renders of its result are **not committed** (personal image); only `job.json` and `metadata.json` are.
- Every step finished on the real models:
  - load;
  - segment (background_color, subject covers 7% of the image);
  - frame to 2048;
  - TripoSG body (balanced profile, 50,000 triangles, 77 s);
  - anatomy (RTMW+MHR in WSL; left and right hands found);
  - two TripoSG hand close-ups (56 s and 59 s);
  - merge (1,114 body faces removed, 2 hand shells added, not watertight);
  - colour, validate, export (GLB 3.0 MB, OBJ 9.5 MB).
- Total wall time **251 s**. Peak VRAM is not recorded by the omni path. TripoSG runs on the same card measured 5.4 to 6.0 GB in the teacher rounds.
- No Windows-specific failure: paths, WSL quoting and the segmenter on the 2048 frame all worked.

## Defect seen (observation)
- The body is good. Each hand close-up, however, became a set of thin fragments spread along the depth axis. Measured after placement:
  - left hand part depth extent **1.71 m**, right **0.46 m**;
  - their crops are about 0.24 m wide.
- The same over-extent is present on human_v1 (contour round, 24 hands): median part depth **0.36 m**, max 0.73 m. A real hand is about 0.25 m or less.
- Hypothesis: the scale comes only from the crop silhouette height, and the depth of a TripoSG close-up is not constrained. This is likely a main reason the teacher rounds gain little. A depth clamp or arbiter gate (for example, rejecting or compressing a part deeper than its crop is wide) is the next experiment. It changes benchmark behaviour, so it should be measured as its own round.
