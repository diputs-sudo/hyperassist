# Gradient Noise Scale-Based Batch Size

## Formula

```
batch_size = int(round(gradient_noise_scale / target_variance))
```

- **`gradient_noise_scale`**: Estimated gradient noise scale (GNS), a measure of stochasticity in the gradient (float, >0).
- **`target_variance`**: Desired variance level for stochastic gradient descent (SGD) updates (float, >0; typical default: `1.0`).
- **`min_batch_size`**: Minimum allowable batch size (default: `1`).

---

## Intuitive Explanation

This formula sets the **training batch size** so that the variance of stochastic gradients remains at a desired, controlled level:

- As **gradient noise increases**, use a **larger batch size** to stabilize updates.
- For **smaller GNS** (less noisy gradients), smaller batches are acceptable and computationally cheaper.
- The ratio keeps SGD “well-behaved” for optimization and generalization.

---

## Theoretical Background / Origin

- Based on the theory and experiments in [Smith et al., 2017 (“Don’t Decay the Learning Rate, Increase the Batch Size”)](https://arxiv.org/abs/1711.00489).
- The **gradient noise scale** quantifies the “effective noise” in stochastic optimization, linking batch size, learning rate, and training dynamics.

---

## Example Calculation

Suppose:
- `gradient_noise_scale = 400`
- `target_variance = 1.0`

Calculation:
- `batch_size = int(round(400 / 1.0)) = 400`

If you want a noisier update (e.g., `target_variance = 2.0`):
- `batch_size = int(round(400 / 2.0)) = 200`

Always clamp to at least `min_batch_size` (usually `1`).

---

## Practical Guidance

- **`gradient_noise_scale`**: Can be estimated empirically from gradient statistics, or from literature for your model/task.
- **`target_variance`**: Typical values are between `0.5` and `2.0`. Smaller values mean less noise (larger batches).
- **Batch size**: Round to the nearest power of 2 for hardware efficiency (e.g., 128, 256, 512).

---

## Python Example

```python
def gns_batch_size(gradient_noise_scale, target_variance=1.0, min_batch_size=1):
    batch = int(round(gradient_noise_scale / target_variance))
    return max(batch, min_batch_size)

# Example usage:
batch = gns_batch_size(400)
print(f"Recommended batch size: {batch}")  # Output: 400
```

---

## Notes & Pitfalls
- **Estimating GNS** requires multiple gradient computations per batch; not always practical in every training loop.
- If unsure, use published values for your architecture as a starting point.
- Don’t set batch size above hardware/VRAM limits—always check memory usage!
- This formula assumes “classic” SGD/Adam; for more advanced optimizers or large-scale distributed training, adjust accordingly.

---
