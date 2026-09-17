# How to Estimate Required Sample Size for Model Training

> Study / prototype — frozen. Estimates training-set size vs accuracy with a power-law fit on `tf_flowers` + MobileNetV2.

## 1. Overview

Answers: "how many images do I need to reach target accuracy?" Trains MobileNetV2 (frozen ImageNet backbone + new head) on subsets `[165, 330, 825, 1651]` of `tf_flowers`, fits `accuracy ≈ a·n^b`, extrapolates.

Status: educational port of the Keras sample-size recipe. No tests, no packaging, results partly hardcoded. Actively maintained: no.

## 2. Method

1. Load `tf_flowers` via `tensorflow_datasets` (90/10 split), resize to 224×224.
2. Augment (flip / rotation / zoom / contrast), frozen MobileNetV2 + GAP + Dropout(0.3) + softmax.
3. Train on increasing subsets, record `train_acc`.
4. Fit power law in log-space (`np.polyfit` on `log(n)`, `log(acc)`), solve for n at target accuracy, plot.

## 3. Repo layout

```
Estimating_required_sample_size_for_model_training.ipynb  # original notebook
estimating.py                                            # import-safe port (helpers + main() guard)
README.md
requirements.txt
```

## 4. Install

```
pip install -r requirements.txt
# tensorflow>=2.16,<2.20, keras>=3,<4, tensorflow-datasets, matplotlib, numpy
# Needs ~2 GB for tf_flowers + ImageNet weights. GPU recommended.
```

## 5. Usage

```bash
# Full run: download tf_flowers, fit power-law curve, train full model
python estimating.py

# Also train on the data subsets (slow: 4 fractions x 5 iters of transfer learning)
python estimating.py --fraction-sweep

# Override training length
python estimating.py --epochs 10 --fine-tune-epochs 20
```

Importing `estimating` as a module is side-effect free (no download, no
training) — execution lives under `main()` + `if __name__ == "__main__"`.

## 6. Expected output

Console: class names, sample counts, per-subset accuracy; a matplotlib log-log plot with fitted curve + extrapolated n for the target. No artifacts saved to disk.

## 7. Limitations / what this is not

- Single dataset/backbone, single seed (42); no confidence intervals.
- Power-law extrapolation is heuristic — breaks under distribution shift.
- Hardcoded `sample_sizes` / `train_acc` in the plotting cell; not a reusable estimator.
- No tests, no CLI args, no pinned run provenance.

## 8. Tests

None. Smoke check = run the script end-to-end (downloads data, ~10–30 min on GPU).

## 9. References

- Keras recipe: https://keras.io/examples/keras_recipes/sample_size_estimate/
- Sample-size review (medical imaging): https://www.researchgate.net/publication/335779941_Sample-Size_Determination_Methodologies_for_Machine_Learning_in_Medical_Imaging_Research_A_Systematic_Review

## 10. License

None declared (no `LICENSE` file). Treat as all-rights-reserved study code; do not reuse commercially without author permission.
