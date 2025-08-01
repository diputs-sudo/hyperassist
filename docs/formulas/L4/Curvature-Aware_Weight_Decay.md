# Curvature-Aware Weight Decay

## Formula

```
weight_decay = α / λ_max(Hessian)
```

- **`λ_max(Hessian)`**: Largest eigenvalue of the Hessian matrix of the loss with respect to model parameters (float, must be >0).
- **`α` (alpha)**: Scaling factor (default: `1.0`; can be tuned).
- **`min_weight_decay`**: Minimum allowed value (e.g., `1e-5`) for safety.

---

## Intuitive Explanation

This formula adjusts **weight decay** (L2 regularization) **inversely with the sharpness of the loss landscape**:

- If the loss surface is *sharp* (large eigenvalue), use *less* weight decay.
- If the loss surface is *flat* (small eigenvalue), use *more* weight decay.
- The goal: Regularize more when the model is in a “flatter”/easier-to-overfit region, and regularize less when near sharp optima.

---

## Theoretical Background / Origin

- Inspired by modern **sharpness-aware minimization** and classical regularization theory.
- See: [Hochreiter & Schmidhuber, 1997](https://www.researchgate.net/publication/14100213_Flat_Minima) (flat minima), and various works on curvature-aware/adaptive regularization ([Smith et al., 2020](https://arxiv.org/abs/2012.07976)).
- The Hessian’s largest eigenvalue reflects local landscape “sharpness,” linking regularization strength to generalization theory.

---

## Example Calculation

Suppose:
- `hessian_max_eig = 25.0`
- `alpha = 1.0`

Calculation:
- `weight_decay = 1.0 / 25.0 = 0.04`

If `hessian_max_eig = 1000.0`, then `weight_decay = 1.0 / 1000.0 = 0.001`

If eigenvalue is very small or zero, use `min_weight_decay` (e.g., `1e-5`).

---

## Practical Guidance

- **`hessian_max_eig`**: Estimate via second-order methods or libraries. For most, use a rough estimate if exact calculation is too expensive.
- **`alpha`**: Tune for your dataset/model; `1.0` is a good start.
- **`min_weight_decay`**: Always set a lower bound to avoid division by zero or instability.
- **Interpretation:** Higher curvature (sharper minima) means model is less prone to overfitting, so you can reduce regularization.

---

## Python Example

```python
def curvature_aware_weight_decay(hessian_max_eig, alpha=1.0, min_weight_decay=1e-5):
    if hessian_max_eig > 0:
        return alpha / hessian_max_eig
    else:
        return min_weight_decay

# Example usage:
wd = curvature_aware_weight_decay(25.0)
print(f"Recommended weight decay: {wd:.4f}")  # Output: 0.0400
```

---

## Notes & Pitfalls
- **Exact Hessian eigenvalue computation is costly** for large models; consider using an approximation.
- For adaptive optimizers (AdamW, LAMB, etc.), effective weight decay may also depend on learning rate—tune jointly.
- This approach assumes larger curvature means less need for regularization, which aligns with some but not all generalization theories, **always validate empirically**.
