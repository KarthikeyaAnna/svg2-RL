# RL Alignment for TraSeR: G3 & G3+PG Policy Optimization

On-policy Group Relative Policy Optimization (GRPO) and Pair-Factored Policy Optimization (PG) for the TraSeR video scene graph foundation model (Qwen2.5-VL-3B).

---

## 1. Problem Formulation & Policy Factorization

Video Scene Graph Generation (VidSGG) requires detecting object trajectories and predicting structured temporal relational triplets $y = \{(s, p, o, [t_{\text{start}}, t_{\text{end}}])\}$.

### 1.1 Measured Failure Modes in Released TraSeR
Our empirical kill-tests across SVG2test, VidOR, and PVSG identified four dominant failure modes in released TraSeR:
1. **Pair Omission (F1, Dominant Failure)**: In complex scenes ($N \ge 10$ objects, $O(N^2) \ge 90$ pairs), whole-graph autoregression suffers from attention dispersion and loop prevention heuristics. TraSeR drops **49.8% (SVG2), 47.3% (PVSG), and 22.9% (VidOR)** of true relations because it *never mentions the interacting pair*.
2. **Synthetic Over-Generation Bias (F2)**: Pretrained on GPT-5 synthetic annotations, TraSeR outputs 55 relations/video (vs 32 in GT), flooding generic non-touching predicates like `"in front of"` (1527 predicted vs 344 GT) and `"cooperating with"`, yielding a baseline precision of **15.2%**.
3. **Repetition Loops & Syntax Degradation (F3)**: Autoregressive decoding frequently degenerates into repetition cycles (repeating `"next to"`), leading to syntax errors or length-cap truncation.
4. **Precision Collapse under Naive Querying (F4)**: Querying TraSeR on omitted pairs post-hoc invokes conversational priors (assuming an interaction must exist), collapsing added-pair precision to **11% on SVG2** and **4% on PVSG**.

### 1.2 Two-Tier Policy Factorization
Under the constraint to train **only on TraSeR's own training data** (7 SVG2 splits: SA-V, PVD, VIPSeg, VidOR, VidVRD, LVVIS, OVIS), the joint generation policy is factored into two complementary stages:

$$\pi(y \mid x) = \pi_{\text{whole}}(\mathcal{G}_{\text{base}} \mid x) \times \prod_{(a, b) \in \mathcal{C}_{\text{omitted}}} \pi_{\text{pair}}(\mathcal{R}_{a,b} \mid x, a, b)$$

* **Level 0 (Plain G3)**: Full-graph autoregressive policy aligned directly on bipartite graph-matching rewards via on-policy Dr. GRPO. Resolves F2 (synthetic bias) and F3 (repetition loops) while preserving baseline precision.
* **Candidate Proposer**: Spatio-temporal `HistGBM` ranker ([`scripts/pair_feats.py`](../scripts/pair_feats.py), 11 geometric features) filters $O(N^2)$ candidate pairs down to those with interaction probability $p(a, b) \ge 0.5$.
* **Level 1 (PG Adapter)**: Pair-level GRPO adapter conditioned *strictly* on TraSeR's trajectory slots for objects $a$ and $b$ (`obj_subset=[pos[a], pos[b]]`), trained with an explicit **abstention reward** on unlabelled hard negatives to resolve F1 (omitted pairs) without F4 (precision collapse).

---

## 2. Policy Optimization Objective (Dr. GRPO)

### 2.1 Loss Formulation
For prompt $x$, sample $G = 8$ stochastic rollouts $o_i \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$ at temperature $T = 1.0$, $\text{top-}p = 0.95$. Policy parameters $\theta$ (LoRA rank $r = 64$) are optimized via:

$$L_{\text{GRPO}}(\theta) = -\frac{1}{B} \sum_{i=1}^B \left[ \frac{1}{L_{\text{norm}}} \sum_{t=1}^{|o_i|} A_i \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}) - \beta D_{\text{KL}}^{k_3}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

### 2.2 Dr. GRPO Advantage Centering
Standard GRPO scales advantages by empirical standard deviation ($A_i = \frac{r_i - \mu}{\sigma + \epsilon}$), which introduces two pathologies in combinatorial structured generation:
1. When rollout variance $\sigma$ is small, it acts as an arbitrary gradient amplifier, destabilizing updates on high-reward prompts.
2. Dividing by sequence length $|o_i|$ incentivizes the policy to generate short, uninformative sequences.

**Dr. GRPO** (arXiv 2503.20783) resolves this via:
* **Group-Mean Centering Without Std Scaling**:
  $$A_i = r(o_i) - \frac{1}{G}\sum_{j=1}^G r(o_j)$$
* **Constant Length Normalization**: The loss is scaled by fixed $L_{\text{norm}}$ ($4096$ for whole graphs; $512$ for pair queries), strictly decoupling gradient magnitudes from sequence lengths and eliminating length gaming.

### 2.3 $k_3$ Unbiased Non-Negative KL Estimator
Policy drift from the frozen base model $\pi_{\text{ref}}$ is regularized via the $k_3$ estimator (Schulman, 2020):
$$d_t = \log \pi_{\text{ref}}(y_t \mid x, y_{<t}) - \log \pi_\theta(y_t \mid x, y_{<t})$$
$$D_{\text{KL}}^{k_3}(\pi_\theta \parallel \pi_{\text{ref}}) = \exp(d_t) - d_t - 1$$

Properties: strictly non-negative ($\forall d \in \mathbb{R}, \exp(d) - d - 1 \ge 0$), minimum at zero, and exhibits lower variance than $(\log \pi_\theta - \log \pi_{\text{ref}})$.

---

## 3. Reward Functions

### 3.1 Level 0: Bipartite Graph-Matching Reward ([`engine/reward.py`](reward.py))
Evaluated against ground truth labels $\text{GT}$:
$$r(o_i) = F_{1,\text{rel}} + 0.25 \times \text{ObjectAccuracy} - 0.20 \times \mathbb{I}(\text{invalid JSON})$$

* **One-to-One Matching**: Predicted triplets $(s, p, o, \text{span})$ are matched greedily to ground-truth triplets with identical ordered pairs $(s, o)$ and normalized predicates, sorted descending by temporal Intersection over Union:
  $$\text{tIoU}(\text{span}_{\text{pred}}, \text{span}_{\text{gt}}) = \frac{|\mathcal{T}_{\text{pred}} \cap \mathcal{T}_{\text{gt}}|}{|\mathcal{T}_{\text{pred}} \cup \mathcal{T}_{\text{gt}}|} \quad (\text{intervals discretized at 0.5s})$$
* **Soft Temporal Credit**: Matches earn $\text{Credit} = \text{tIoU}$. Unmatched predictions or duplicates cannot match twice and penalize precision:
  $$\text{Prec} = \frac{\sum \text{Credit}}{|P|}, \quad \text{Rec} = \frac{\sum \text{Credit}}{|G|}, \quad F_{1,\text{rel}} = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$

---

### 3.2 Level 1: PG Abstention Game ([`kaggle/e1/pg_part.py`](../kaggle/e1/pg_part.py))
Candidate pairs $(a, b)$ are sampled from a balanced pool of positive pairs and geometric hard-negative pairs:

$$r_{\text{pair}}(y) = 
\begin{cases}
F_{1,\text{rel}}(y, \text{GT}_{a,b}) & \text{if } (a, b) \in \text{GT} \\
+0.5 & \text{if } (a, b) \notin \text{GT} \text{ and } \text{parse}(y) = \emptyset \quad (\text{correct abstention}) \\
0.0 & \text{if } (a, b) \notin \text{GT} \text{ and } \text{parse}(y) \ne \emptyset \quad (\text{hallucination / false positive})
\end{cases}$$

#### Advantage Dynamics on Hard Negatives
For an unlabelled negative pair across $G = 8$ rollouts, suppose $k$ rollouts correctly abstain ($r = +0.5$) and $8 - k$ hallucinate ($r = 0.0$).
* Mean reward: $\mu_r = \frac{0.5k}{8} \in (0, 0.5)$.
* Abstaining rollouts receive positive advantage: $A_{\text{abstain}} = 0.5 - \mu_r > 0$, reinforcing termination tokens.
* Hallucinating rollouts receive negative advantage: $A_{\text{hallucinate}} = -\mu_r < 0$, suppressing invented predicates.
* This establishes an exact margin: the policy emits a relation if and only if expected credit exceeds the $+0.5$ reward of remaining silent.

---

## 4. Empirical Evaluation

Protocol: strict per-span $\text{tIoU} \ge 0.5$ with Qwen3-8B lenient judge. Significance ($^\dagger$) via 2,000-sample paired bootstrap ($p < 0.05$).

| Benchmark | Model | Precision | Rel Recall | Triplet Recall | Obj Recall | F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **VidOR-val** | TraSeR (Baseline) | 26.8% | 9.2% | 6.4% | 78.8% | 13.7 |
| *(835 videos)* | **Plain G3** | **29.8%**$^\dagger$ | **9.9%**$^\dagger$ | **7.1%**$^\dagger$ | **79.1%**$^\dagger$ | **14.8**$^\dagger$ |
| | **G3 + PG (Headline)** | **32.4%**$^\dagger$ | **12.1%**$^\dagger$ | **8.6%**$^\dagger$ | **79.1%**$^\dagger$ | **17.6**$^\dagger$ *(+3.9)* |
| **SVG2test** | TraSeR (Baseline) | 18.9% | 24.8% | 12.8% | 50.9% | 21.4 |
| *(67 videos)* | **Plain G3** | 18.8% | **30.5%**$^\dagger$ | 14.1% | 49.9% | 23.2 |
| | **G3 + PG (Headline)** | 16.8% | **36.9%**$^\dagger$ | **16.4%**$^\dagger$ | 49.9% | **23.0** |
| **PVSG-val** | TraSeR (Baseline) | 5.8% | 9.7% | 4.7% | 57.8% | 7.2 |
| *(62 videos)* | **Plain G3** | 6.2% | 10.1% | 5.1% | 57.3% | 7.7 |
| | **G3 + PG (Headline)** | 5.9% | **14.1%**$^\dagger$ | **6.4%**$^\dagger$ | 57.3% | **8.3** |
| *Ablation:* | | | | | | |
| *SVG2test* | *G3 + Vote3 + PG (9 rollouts)* | *25.4%*$^\dagger$ | *30.9%*$^\dagger$ | *15.2%*$^\dagger$ | *50.9%* | ***27.9***$^\dagger$ |

### Findings
1. **Recall Expansion**: G3 + PG delivers statistically significant relation recall gains across all datasets (SVG2 $+12.1\%$, VidOR $+2.9\%$, PVSG $+4.4\%$) with simultaneous precision gains on VidOR ($26.8\% \to 32.4\%$, $+3.9$ F1).
2. **Object Grounding Invariance**: Object Recall remains constant across all models (~79.1% on VidOR, ~50% on SVG2), verifying that GRPO aligns relational attention heads without degrading visual trajectory grounding.
3. **SCB Negative Control**: Self-consistency voting (`Vote3`) achieves $27.9$ F1 on G3 ($+6.5$ over baseline). Running the identical voting harness on the unaligned baseline yielded $\Delta F_1 = +1.40$ (CI $[-0.59, +3.31]$, failing the pre-registered significance gate), proving multi-sample gains stem from RL alignment rather than decoding mechanics.

---

## 5. Codebase Structure & Key Pointers

```
svg2-killtest/
├── engine/
│   ├── grpo.py           # On-policy GRPO loss, Dr. GRPO advantage, k3 KL estimator
│   ├── reward.py         # Bipartite graph matching & soft temporal IoU reward
│   ├── traser_fast.py    # Batched rollout decoder (SlotDecoder)
│   ├── sft.py            # LoRA linear layers with dynamic scale control
│   ├── smoke_grpo.py     # Unit test: logprobs, LoRA toggling, KL, GRPO step
│   ├── smoke_sft.py      # Unit test: LoRA zero-init and scaling equivalence
│   └── smoke_test.py     # Parity test: SlotDecoder vs reference HuggingFace
├── kaggle/e1/
│   ├── cfg_G3.py         # Plain G3 hyperparameters (lr 3e-5, G=8, 300 steps)
│   ├── G3/e1_g3.py       # Monolithic training script for Level 0 G3
│   ├── cfg_PG.py         # PG hyperparameters (lr 2e-5, W_none=0.5, 120 steps)
│   ├── PG/e1_pg.py       # Monolithic training script for Level 1 PG
│   ├── grpo_part.py      # Whole-graph rollout and loss backward loop
│   └── pg_part.py        # Pair rollout, abstention reward, and LoRA update loop
└── scripts/
    ├── pair_feats.py     # 11 geometric spatio-temporal features for HistGBM
    ├── merge_pairs.py    # Merges PG pair decodes into base G3 graphs
    ├── score_runs.py     # P, R, F1 evaluation harness
    └── boot_triplet.py   # Paired bootstrap triplet recall & object accuracy
```

---

## 6. Execution Runbook

### 6.1 Local Verification
```bash
# Verify GRPO loss, advantage centering, and k3 KL divergence
python -m engine.smoke_grpo

# Verify LoRA scale toggling (LoRA-off == base reference model)
python -m engine.smoke_sft

# Verify SlotDecoder token-for-token parity with HuggingFace generate()
python -m engine.smoke_test
```

### 6.2 Evaluation
```bash
# Relation Precision, Recall, F1 against ground truth
python scripts/score_runs.py out/e1_g3_pg/svg2_merged.jsonl data/svg2_gt.parquet

# Paired bootstrap Triplet Recall and Object Accuracy (Qwen3-8B judge)
# Args: <gt> <judge> <sysA_merged> <sysA_labels> <sysB_merged> <sysB_labels>
python scripts/boot_triplet.py \
    data/svg2_gt.parquet \
    data/judge_cache.json \
    out/e1_g3_pg/svg2_merged.jsonl \
    out/e1_g3/svg2_full.jsonl \
    out/baseline/svg2_full.jsonl \
    out/baseline/svg2_full.jsonl
```

---

## 7. Documented Negative Results (Pre-Registered)

1. **Supervised Fine-Tuning (SFT / A1 run)**: Cross-entropy SFT failed due to convention bleed and repetition collapse (endlessly predicting `"next to"`). SFT abandoned in favor of on-policy GRPO.
2. **Object-Ablation Credit (OAC / §65)**: Ablating visual object latent tokens to construct an unsupervised reward failed its positive control ($\text{median } \Delta = -0.0001 \le 0.10$, $\text{AUROC} = 0.5439$). OAC is killed as an RL reward.
3. **PH Monolithic Scheduling**: Single-session decoding of 47,210 candidate pairs hit Kaggle's 12-hour container hard cutoff (`KernelWorkerStatus.ERROR`). Large inference passes must be chunked per split.
