# Weight Decay Calculation Formula

## Formula

```
weight_decay = 1e-5 * compute_factor
```

- `compute_factor`: Adjusts for the power of your compute resources.  
  Use `1.0` for standard GPUs/CPUs, lower for slow machines (e.g., 0.5), or higher for extremely fast compute (rare).

## Explanation

**Weight decay** (sometimes called L2 regularization) helps prevent overfitting by penalizing large weights during training.  
This formula sets a **sensible default** and lets you adjust if your compute setup is unusually limited or powerful.

- `1e-5` is a widely used standard value for AdamW, Adam, and many modern optimizers.
- Increasing `compute_factor` slightly increases regularization, useful if you’re running lots of experiments or with massive models.
- Decreasing `compute_factor` can help when you have very limited compute and want to speed up training.

> **Example:**  
> - Standard machine: `compute_factor = 1.0` -> `weight_decay = 1e-5`
> - Slow/limited device: `compute_factor = 0.5` -> `weight_decay = 5e-6`
> - Large scale/robust setup: `compute_factor = 2.0` -> `weight_decay = 2e-5`

## Python Code Example

```python
def compute_weight_decay(compute_factor=1.0):
    return 1e-5 * compute_factor

# Example usage:
print(compute_weight_decay())      # Output: 1e-5
print(compute_weight_decay(0.5))   # Output: 5e-6
print(compute_weight_decay(2.0))   # Output: 2e-5
```

## Notes
- For most projects, `1e-5` is a robust default.
- Use `compute_factor` if you need to scale for very different hardware.
- Tune further if your validation loss plateaus or overfits.