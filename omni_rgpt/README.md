# Omni-RGPT-style Qwen2.5-VL-3B: results

Base Qwen2.5-VL-3B + 40 learnable token marks (Omni-RGPT style: each mark is added to the visual tokens under one object's mask)
+ LoRA SFT. Trained only on TraSeR's own training splits (SVG2). Same test sets and scorer as TraSeR.

- **v1**: trained on ~53k videos, almost all PVD and SA-V (1.1k VidOR).
- **v2**: v1 + 44.3k more videos, including VidOR.

## Training data

| Training videos | PVD | SA-V | VidOR | LV-VIS | VIPSeg | VidVRD | OVIS | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TraSeR | ~590k | 42.9k | 7.0k | 3.1k | 2.8k | 0.8k | 0.6k | **~646k** |
| **v1** (ours) | ~34.4k | ~18.2k | 1.1k | 0 | 0 | 0 | 0 | **~53.3k** |
| **v2** (ours) | ~46.6k | ~42.5k | ~6.9k | 0 | 0 | 0.76k | 0 | **~96.8k** |
| % of TraSeR used by v2 | ~7.9% | ~99% | ~99% | 0% | 0% | ~95% | 0% | **~15%** |

These are unique videos, rounded. v2 = v1 weights trained for one more epoch on the new videos.

## Benchmark results

Relation F1 (%). Graph output plus the pair-question answers, which is our full pipeline.

| | Relation F1 (judge): SVG2test-100 | VidOR | PVSG | Relation F1 (exact): SVG2test-100 | VidOR | PVSG |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen, zero-shot | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **v1** (ours) | **25.3** | 17.4 | 10.4 | 14.3 | 4.6 † | 4.6 |
| **v2** (ours) | 24.0 | **32.4** † | 7.3 † | **14.4** | **21.3** † | 1.0 † |
| TraSeR (released) | 22.2 | 18.0 | **10.9** | 13.4 | 5.5 | **5.2** |

**Bold** = best in the column. † = significantly different from TraSeR (paired per-video bootstrap, 95% CI excludes 0).

### Precision / Recall / F1

Same outputs as above: graph + pair answers. Values are P / R / F1 in %. † on R or F1 = significantly different from TraSeR
(paired per-video bootstrap, 95% CI excludes 0).

**Exact**

| | SVG2test-100 | VidOR | PVSG |
|---|---|---|---|
| v1 (ours) | 11.8 / 18.1 / 14.3 | 8.0 / 3.2 † / 4.6 † | 3.5 / 6.7 / 4.6 |
| v2 (ours) | **11.9** / **18.3** / **14.4** | **24.3** / **18.9** † / **21.3** † | 0.7 / 2.1 † / 1.0 † |
| TraSeR (released) | 11.1 / 16.7 / 13.4 | 10.1 / 3.8 / 5.5 | **4.1** / **7.2** / **5.2** |

**Judge**

| | SVG2test-100 | VidOR | PVSG |
|---|---|---|---|
| v1 (ours) | **21.1** / **31.6** † / **25.3** | 34.6 / 11.6 / 17.4 | 8.6 / 13.2 / 10.4 |
| v2 (ours) | 20.0 / 30.0 / 24.0 | **38.9** / **27.7** † / **32.4** † | 4.9 / **13.8** / 7.3 † |
| TraSeR (released) | 18.8 / 27.1 / 22.2 | 37.2 / 11.8 / 18.0 | **9.3** / 13.2 / **10.9** |

### Graph output only (no pair questions)

| | Judge: SVG2test-100 | VidOR | PVSG | Exact: SVG2test-100 | VidOR | PVSG |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 23.8 | 13.6 † | 9.4 | 13.7 | 3.7 † | 4.3 |
| v2 | 22.2 | 26.8 † | 4.8 † | 13.8 | 17.3 † | 0.6 † |
| TraSeR | 22.2 | 18.0 | 10.9 | 13.4 | 5.5 | 5.2 |

## How it is scored

- **Test sets:** SVG2test-100 (67 VIPSeg + 33 SA-V videos), VidOR-val (835), PVSG-val (62).
- **Exact:** same subject and object, the same predicate after normalisation, and temporal IoU ≥ 0.5 for each span.
  Exact duplicates are merged and counts are pooled over all videos.
- **Judge:** Gemini 3.8 Flash was used as the judge.
- The zero-shot row is from the earlier run with the same exact scorer.

