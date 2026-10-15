# TraSeR with training-free pair querying

Results on SVG2test, PVSG-val and VidOR-val. Full report: [`report/traser_salience.pdf`](report/traser_salience.pdf).

## 1. Summary

Training-free post-processing on released TraSeR (Qwen2.5-VL-3B); relation P/R/F1.
SVG2test: +40 contact-ranked pair queries, R 24.7 → **43.3**† (F1 n.s.); contact filter, F1 20.1 → **24.3**†.
VidOR-val: the configuration pre-registered from SVG2test is null; an exploratory one (20 pairs + reverse-direction completion)
gives F1 13.9 → **19.1**†, but contact ranking does not beat random pairs. PVSG-val: recall up, F1 down†. The contact filter
helps SVG2test only (PVSG n.s., VidOR F1 −2.2†). No single configuration improves all three.

## 2. Where TraSeR loses relations

Table 1: Outcome of every GT relation for released TraSeR (% of GT relations; columns sum to 100). Matching as in Section 3;
temporal IoU on the merged span.

| | SVG2test | PVSG-val | VidOR-val |
|---|---:|---:|---:|
| Found | 25.1 | 6.7 | 5.7 |
| Pair never mentioned | 49.8 | 47.3 | 22.9 |
| Pair stated only in the other direction | 7.3 | 4.0 | 27.8 |
| Right pair, wrong predicate | 9.2 | 22.5 | 22.3 |
| Right relation, wrong time span | 1.7 | 17.5 | 21.3 |
| Object not in TraSeR's input (cap / empty mask) | 7.0 | 2.0 | 0.0 |
| **Relations per video, TraSeR vs GT** | 55 vs 32 | 62 vs 16 | 12 vs 36 |

## 3. Setup

**Data.** SVG2test: 67 VIPSeg videos, 66 scored (33 SA-V videos licence-gated, not run). PVSG-val: 62 videos, plus a 27-video
clean subset (35/40 VidOR-sourced PVSG videos are in TraSeR's training set). VidOR-val: 835 videos. GT object masks are given
(SVG2 protocol).

**Systems.** All operate on TraSeR's output and the given GT masks; TraSeR's weights are unchanged.

- **Contact filter.** For every predicted relation (s, p, o), take the two objects' masks on the sampled frames (4× downsampled),
  dilate one by 2 cells, and test overlap with the other. If they never overlap in any frame, the relation is removed.
  Rationale: SVG2test annotators label 15.4% of touching vs 0.8% of non-touching pairs.
- **+K salient.** (1) List object pairs that appear in no predicted relation. (2) Score each by the fraction of co-visible frames
  in which the dilated masks touch; keep the top K. (3) For each pair, run TraSeR again with only those two objects as input
  (same prompt) and parse the relations it returns. (4) Map them back to the original object IDs and append them to the graph;
  exact duplicates are merged. Rationale: 50% (SVG2test) / 47% (PVSG) of all GT relations are on pairs TraSeR never mentions
  (Table 1).
- **+20 random.** Steps (1), (3) and (4) with 20 pairs drawn uniformly from the unmentioned pairs: same number of extra queries,
  no contact ranking (control).
- **Filter + 20 salient.** Contact filter, then +20 salient.
- **Both directions.** For each relation (s, p, o) whose predicate has a known inverse (e.g. above ↔ below, in front of ↔ behind,
  on ↔ under, inside ↔ contains) or is symmetric (next to, near, beside), add (o, p⁻¹, s) with the same time spans if it is not
  already present. Rationale: VidOR labels 81% of pairs in both directions, TraSeR 12%.

VidOR-val config (filter + 20 salient) was fixed on SVG2test before any VidOR run.

**Metric.** Match = same ordered pair, predicate equivalent under a Qwen3-8B judge (SVG2's five categories; stand-in for
GPT-4o-mini), temporal IoU ≥ 0.5 per span; exact duplicates merged. †: paired per-video bootstrap (2000), 95% CI excludes 0.

## 4. Results

Table 2: Relation P / R / F1 (%). † significantly different from TraSeR (paired per-video bootstrap, 95% CI excludes 0).
**Bold**: best per column. \* pre-registered for VidOR-val.

| System | SVG2 P | SVG2 R | SVG2 F1 | PVSG P | PVSG R | PVSG F1 | PVSG-clean P | PVSG-clean R | PVSG-clean F1 | VidOR P | VidOR R | VidOR F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TraSeR | 16.9 | 24.7 | 20.1 | 5.8 | 10.2 | 7.4 | 5.3 | 8.7 | 6.6 | 27.2 | 9.3 | 13.9 |
| Contact filter | **27.4**† | 21.8† | **24.3**† | **6.8**† | 9.5 | **7.9** | **6.4** | 7.3 | **6.8** | **30.6**† | 7.2† | 11.7† |
| +20 random (control) | 10.3† | 27.7† | 15.0† | 2.8† | 11.7 | 4.5† | 2.1† | 8.3 | 3.4† | 20.0† | 12.0† | 15.0† |
| +20 salient | 14.0† | 35.8† | 20.1 | 3.8† | 15.5† | 6.2 | 2.9† | 11.7 | 4.7 | 25.3† | 11.2† | 15.5† |
| +40 salient | 12.3† | **43.3**† | 19.2 | 2.9† | **16.9**† | 5.0† | 2.3† | **13.3**† | 3.9† | 25.3† | 11.3† | 15.6† |
| Filter + 20 salient\* | 16.9 | 32.2† | 22.1 | 4.1† | 15.4† | 6.4 | 3.2† | 11.7 | 5.0 | 27.2 | 9.1 | 13.6 |
| Both directions | 10.1† | 26.1† | 14.6† | 3.7† | 10.4 | 5.5† | 3.5† | 8.7 | 5.0† | 24.5† | 13.1† | 17.1† |
| +20 salient + both directions | 8.9† | 38.7† | 14.4† | 2.6† | 16.0† | 4.5† | 2.1† | 12.1 | 3.6† | 23.7† | **16.0**† | **19.1**† |

Video counts: SVG2test 66, PVSG-val 62, PVSG clean 27, VidOR-val 835.

## 5. Next step

GRPO on TraSeR's own training data (same seven SVG2 splits) with a graph-matching reward (relation F1 with soft temporal IoU,
object accuracy, format), one configuration for all three benchmarks; control: compute-matched SFT on the same videos. Gated on a
pre-check: best-of-8 vs greedy reward on training videos.
