# GRPO Training Recommendations - VLM-R1

**Date**: 2025-11-16
**Analysis**: Based on GRPO_analysis results and literature review
**Current Status**: Trained until step 1500 with checkpoint-1400/1500 showing best accuracy

---

## Executive Summary

Current training shows **accuracy-reasoning trade-off**: checkpoint-1400 has best accuracy (44.92%) but lower reasoning quality (F1=0.4264), while checkpoint-300/700/1100 have better reasoning (F1~0.50) but lower accuracy (~29-39%). The model is becoming too conservative (high precision, low recall, fewer reasoning steps).

## Key Findings from Analysis

### Performance Metrics
- **Best Accuracy**: checkpoint-1400 (44.92%) → +12.7% over baseline
- **Best Reasoning**: checkpoint-300 (F1=0.5071) → +5.6% over baseline
- **Issue**: Models generate fewer steps over training (3.77→3.0→2.85)
- **Conservatism**: Precision increases (96-98%) but recall drops (35%→27%)

### Training Trajectory Issues
1. **Early phase (100-300)**: Good reasoning learning, accuracy drops initially
2. **Mid phase (400-900)**: Stabilization with balance recovery
3. **Late phase (1000-1500)**: Over-optimization for accuracy, reasoning quality degrades

---

## Recommendations

### Option A: Resume from checkpoint-1100 (RECOMMENDED)
**Rationale**: Best balance between accuracy (39.66%) and reasoning (F1=0.502)

```bash
# Update resume_train_vqa_deepspeed.sh:
CHECKPOINT_PATH="${PROJECT_ROOT}/output/.../checkpoint-1100"
```

**Why not 1400/1500?**
- Already over-optimized for accuracy at expense of reasoning depth
- Generates too few steps (2.85-3.0 vs optimal 3.76)
- Lower recall indicates model is too conservative

### Option B: Resume from checkpoint-700 (ALTERNATIVE)
**Rationale**: Highest reasoning quality (F1=0.5063) with decent accuracy (38.92%)

Best if reasoning quality is priority over raw accuracy.

---

## Hyperparameter Changes

### Current vs Recommended

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **learning_rate** | 3e-6 | **2e-6** | More conservative to prevent over-optimization |
| **num_generations** | 2 | **4** | Increase group size for better gradient signal & less variance |
| **reward_weights** | 2.0, 3.0, 1.0 | **2.0, 2.5, 1.5** | Reduce accuracy dominance, boost reasoning |
| **gradient_accum** | 2 | **4** | Larger effective batch for stability |
| **per_device_batch** | 4 | **2** | Compensate for increased num_generations memory |

### Key Changes Explained

**1. num_generations: 2→4**
- Literature recommends 4-8 for VLM GRPO (current 2 is too low)
- Larger groups = smoother gradients, less variance
- Prevents reward collapse by providing better baseline

**2. reward_weights: (2.0, 3.0, 1.0) → (2.0, 2.5, 1.5)**
- Current setup over-weights accuracy (3.0) causing reasoning degradation
- Boost reasoning weight from 1.0→1.5 (+50%)
- Reduce accuracy from 3.0→2.5 to allow more reasoning exploration

**3. learning_rate: 3e-6 → 2e-6**
- More conservative updates to prevent overfitting to accuracy metric
- Literature shows 1e-6 to 5e-6 range; we're in middle-low end

**4. Batch size adjustments**
- Reduce per_device from 4→2 to accommodate 2x num_generations
- Increase gradient_accum 2→4 to maintain effective batch size (32)
- **Total completions per step**: 32 × 4 = 128 (was 64)

---

## Implementation Steps

1. **Update resume_train_vqa_deepspeed.sh**:
   ```bash
   CHECKPOINT_PATH=".../checkpoint-1100"  # or checkpoint-700
   NUM_GENERATIONS=4
   LEARNING_RATE=2e-6
   PER_DEVICE_BATCH=2
   GRADIENT_ACCUM=4
   # Update reward weights line 158
   --reward_weights 2.0 2.5 1.5 \
   ```

2. **Monitor these metrics**:
   - Average steps per prediction (target: 3.5-4.0)
   - Recall metric (target: >0.30)
   - Match F1 (target: maintain >0.48)
   - Accuracy (target: >43%)

3. **Stop conditions**:
   - If accuracy plateaus without reasoning improvement
   - If steps drop below 3.0 average
   - If recall drops below 0.25

---

## Alternative: Curriculum Learning Approach

If simple hyperparameter tuning doesn't work, consider:

1. Resume from checkpoint-300 (best reasoning)
2. Gradually increase accuracy weight over training
3. Start: (2.0, 2.0, 1.5) → End: (2.0, 2.5, 1.2)

---

## References

- GRPO Analysis Report: `GRPO_analysis/ANALYSIS_REPORT.md`
- Literature: GRPO best practices (2024-2025)
  - Group size 4-8 recommended for VLMs
  - KL regularization critical for collapse prevention
  - Larger groups reduce variance
