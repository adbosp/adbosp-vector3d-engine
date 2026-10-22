# Benchmark report

## Environment

```json
{
 "os": "Windows-10-10.0.26300-SP0",
 "wsl": false,
 "python": "3.10.11",
 "cpu": "Intel64 Family 6 Model 191 Stepping 2, GenuineIntel",
 "cpu_count": 16,
 "ram_gb": 31.833328247070312,
 "disk_free_gb": 86.9213981628418,
 "gpus": [
  {
   "name": "NVIDIA GeForce RTX 3060",
   "vram_total_mb": 12288.0,
   "vram_free_mb": 10539.0,
   "driver": "616.92",
   "driver_cuda": null
  }
 ],
 "torch": {
  "torch": null,
  "note": "torch not importable in C:\\adbosp-vector3d-engine\\.venv\\Scripts\\python.exe"
 },
 "tools": {
  "git": true,
  "nvcc": true,
  "cl": false,
  "gcc": false,
  "blender": false,
  "wsl": true
 },
 "models": {
  "triposg": {
   "vram_gb": 8,
   "os": "any (not stated)",
   "license": "MIT",
   "verdict": "fits_on_paper"
  },
  "hunyuan3d-2mini/2mv (shape)": {
   "vram_gb": 6,
   "os": "win/mac/linux",
   "license": "Tencent Hunyuan community (EU/UK/KR excluded)",
   "verdict": "fits_on_paper"
  },
  "hunyuan3d-2.1 (shape)": {
   "vram_gb": 10,
   "os": "win/mac/linux",
   "license": "Tencent Hunyuan community (EU/UK/KR excluded)",
   "verdict": "fits_on_paper"
  },
  "hunyuan3d-2.1 (shape+PBR)": {
   "vram_gb": 29,
   "os": "win/mac/linux",
   "license": "Tencent Hunyuan community (EU/UK/KR excluded)",
   "verdict": "does_not_fit (29 GB needed, 12.0 GB present)"
  },
  "trellis.2": {
   "vram_gb": 24,
   "os": "linux",
   "license": "MIT",
   "verdict": "does_not_fit (24 GB needed, 12.0 GB present); Linux-only, use WSL2 or cloud"
  },
  "pixal3d": {
   "vram_gb": null,
   "os": "not stated (TRELLIS.2 stack)",
   "license": "MIT",
   "verdict": "unknown: no published VRAM figure, must be measured"
  }
 },
 "notes": [
  "MSVC cl.exe not on PATH: CUDA extensions on Windows need the VS Build Tools developer prompt"
 ]
}
```

## Real benchmark: triposg (balanced, seed 42)

| Object | Status | Time (s) | Peak VRAM (MB) | Ref IoU A → B | Held-out IoU A → B | Chamfer-L1 A → B | F@1% A → B | F@2% A → B | Normal cons. A → B |
|---|---|---|---|---|---|---|---|---|---|
| crate | success | 88.7508 | 5746.6875 | 0.5702 → 0.9887 | 0.4214 → 0.9718 | 0.1131 → 0.0312 | 0.1198 → 0.5993 | 0.1652 → 0.6975 | 0.9884 → 0.9495 |
| chair | success | 74.2239 | 5668.3164 | 0.6326 → 0.9789 | 0.7651 → 0.9114 | 0.0151 → 0.0087 | 0.5450 → 0.7644 | 0.7451 → 0.8646 | 0.9558 → 0.9130 |
| mug | success | 94.6195 | 6217.4990 | 0.9797 → 0.9975 | 0.9764 → 0.9722 | 0.0083 → 0.0089 | 0.6409 → 0.5710 | 0.9999 → 0.9998 | 0.9852 → 0.9830 |
| character | success | 65.1041 | 5570.6880 | 0.9273 → 0.9939 | 0.8632 → 0.8478 | 0.0078 → 0.0074 | 0.8028 → 0.7790 | 0.9658 → 0.9579 | 0.9983 → 0.9970 |
| lamp | success | 61.9922 | 5625.4263 | 0.9361 → 0.9862 | 0.9577 → 0.9574 | 0.0063 → 0.0076 | 0.7421 → 0.7564 | 0.9887 → 0.9214 | 0.9466 → 0.9520 |
| bracket | success | 62.8706 | 5623.1577 | 0.7096 → 0.9948 | 0.6211 → 0.9471 | 0.0333 → 0.0058 | 0.3800 → 0.9253 | 0.6253 → 0.9658 | 0.9962 → 0.8727 |

## Experiment 001: silhouette refinement on a simulated baseline

Baseline: **SIMULATED_BASELINE (deformed ground truth, not AI output)**. Dataset `synthetic_v1`, 6 objects × 3 seeds = 18 runs. Parameters: `{"iterations": 40, "step": 0.6, "rim_cos": 0.35, "diffusion": 12, "max_px": 6.0, "patience": 4, "smooth": 3, "blur": 4}`.

| Object | Seed | Ref IoU before → after | Held-out IoU before → after | Chamfer-L1 before → after | F@1% before → after | F@2% before → after | Normal cons. before → after | Gates |
|---|---|---|---|---|---|---|---|---|
| crate | 0 | 0.9230 → 0.9975 | 0.9081 → 0.9722 | 0.0197 → 0.0099 | 0.2703 → 0.7599 | 0.5666 → 0.8634 | 0.9969 → 0.9962 | accepted |
| crate | 1 | 0.8211 → 0.9958 | 0.9126 → 0.9845 | 0.0207 → 0.0062 | 0.2279 → 0.8813 | 0.5310 → 0.9707 | 0.9970 → 0.9951 | accepted |
| crate | 2 | 0.9229 → 0.9939 | 0.9256 → 0.9708 | 0.0177 → 0.0076 | 0.2366 → 0.8017 | 0.6125 → 0.9200 | 0.9966 → 0.9953 | accepted |
| chair | 0 | 0.6287 → 0.9823 | 0.7712 → 0.8888 | 0.0161 → 0.0098 | 0.3853 → 0.7094 | 0.6826 → 0.8307 | 0.9888 → 0.9776 | accepted |
| chair | 1 | 0.6686 → 0.9896 | 0.8130 → 0.9417 | 0.0141 → 0.0082 | 0.3570 → 0.7841 | 0.7872 → 0.8932 | 0.9886 → 0.9813 | accepted |
| chair | 2 | 0.7052 → 0.9868 | 0.8114 → 0.9023 | 0.0114 → 0.0088 | 0.5132 → 0.7498 | 0.9014 → 0.8655 | 0.9885 → 0.9754 | accepted |
| mug | 0 | 0.8817 → 0.9961 | 0.8857 → 0.9045 | 0.0149 → 0.0139 | 0.3610 → 0.4467 | 0.7734 → 0.7582 | 0.9959 → 0.9940 | accepted |
| mug | 1 | 0.8493 → 0.9976 | 0.9229 → 0.9437 | 0.0136 → 0.0108 | 0.3927 → 0.5484 | 0.8321 → 0.9088 | 0.9961 → 0.9948 | accepted |
| mug | 2 | 0.9013 → 0.9964 | 0.8944 → 0.9181 | 0.0133 → 0.0118 | 0.3758 → 0.5005 | 0.8296 → 0.8577 | 0.9959 → 0.9949 | accepted |
| character | 0 | 0.7743 → 0.9922 | 0.8108 → 0.8342 | 0.0141 → 0.0085 | 0.4401 → 0.6752 | 0.7679 → 0.9421 | 0.9963 → 0.9931 | accepted |
| character | 1 | 0.7986 → 0.9924 | 0.8235 → 0.9104 | 0.0122 → 0.0066 | 0.5224 → 0.8000 | 0.8712 → 0.9899 | 0.9963 → 0.9917 | accepted |
| character | 2 | 0.8138 → 0.9919 | 0.7833 → 0.8298 | 0.0141 → 0.0086 | 0.3814 → 0.6863 | 0.7719 → 0.9237 | 0.9962 → 0.9926 | accepted |
| lamp | 0 | 0.6140 → 0.9875 | 0.7914 → 0.8907 | 0.0153 → 0.0100 | 0.4383 → 0.6320 | 0.6943 → 0.8670 | 0.9810 → 0.9702 | accepted |
| lamp | 1 | 0.5960 → 0.9886 | 0.7224 → 0.8505 | 0.0139 → 0.0111 | 0.6151 → 0.6028 | 0.7967 → 0.8299 | 0.9811 → 0.9682 | accepted |
| lamp | 2 | 0.6457 → 0.9875 | 0.7886 → 0.8985 | 0.0133 → 0.0101 | 0.4740 → 0.6465 | 0.7637 → 0.8594 | 0.9812 → 0.9680 | accepted |
| bracket | 0 | 0.8684 → 0.9929 | 0.8675 → 0.9298 | 0.0139 → 0.0067 | 0.4037 → 0.8564 | 0.7788 → 0.9450 | 0.9938 → 0.9913 | accepted |
| bracket | 1 | 0.7488 → 0.9962 | 0.8507 → 0.9617 | 0.0186 → 0.0053 | 0.3525 → 0.9317 | 0.6162 → 0.9814 | 0.9939 → 0.9920 | accepted |
| bracket | 2 | 0.8480 → 0.9954 | 0.8743 → 0.9531 | 0.0148 → 0.0054 | 0.3177 → 0.9331 | 0.7411 → 0.9816 | 0.9937 → 0.9864 | accepted |

**Means over all runs**

| Metric | Before | After | Runs improved |
|---|---|---|---|
| Ref IoU | 0.7783 | 0.9923 | 18/18 |
| Held-out IoU | 0.8421 | 0.9159 | 18/18 |
| Chamfer-L1 | 0.0151 | 0.0089 | 18/18 |
| F@1% | 0.3925 | 0.7192 | 17/18 |
| F@2% | 0.7399 | 0.8993 | 16/18 |
| Normal cons. | 0.9921 | 0.9866 | 0/18 |

**Runs where a metric got worse**

- lamp seed 1: F@1% 0.6151 → 0.6028
- chair seed 2: F@2% 0.9014 → 0.8655
- mug seed 0: F@2% 0.7734 → 0.7582
- crate seed 0: Normal cons. 0.9969 → 0.9962
- crate seed 1: Normal cons. 0.9970 → 0.9951
- crate seed 2: Normal cons. 0.9966 → 0.9953
- chair seed 0: Normal cons. 0.9888 → 0.9776
- chair seed 1: Normal cons. 0.9886 → 0.9813
- chair seed 2: Normal cons. 0.9885 → 0.9754
- mug seed 0: Normal cons. 0.9959 → 0.9940
- mug seed 1: Normal cons. 0.9961 → 0.9948
- mug seed 2: Normal cons. 0.9959 → 0.9949
- character seed 0: Normal cons. 0.9963 → 0.9931
- character seed 1: Normal cons. 0.9963 → 0.9917
- character seed 2: Normal cons. 0.9962 → 0.9926
- lamp seed 0: Normal cons. 0.9810 → 0.9702
- lamp seed 1: Normal cons. 0.9811 → 0.9682
- lamp seed 2: Normal cons. 0.9812 → 0.9680
- bracket seed 0: Normal cons. 0.9938 → 0.9913
- bracket seed 1: Normal cons. 0.9939 → 0.9920
- bracket seed 2: Normal cons. 0.9937 → 0.9864
