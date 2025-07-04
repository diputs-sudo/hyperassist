# Generic Learning Rate Formula

## Formula

```
lr = 0.001 / (log10(dataset_size) + 1) / time_factor
```

- `dataset_size` is the number of samples in your dataset.
- `time_factor` (optional) lets you train faster (`<1.0` for “rush”, `>1.0` for “safer/longer” training).

## Explanation

This is a **generic heuristic** for learning rate, useful for a wide variety of model types and dataset sizes:

- As dataset size increases, the recommended learning rate **decreases**. This stabilizes training on large datasets.
- The `time_factor` parameter lets you fine tune for your hardware or patience—smaller means more aggressive, larger means safer.
- This rule is inspired by practical experience and serves as a safe default when you don’t have a domain specific formula.

> **Example:**  
> - `dataset_size = 10,000`, `time_factor = 1.0`:  
>   `lr = 0.001 / (log10(10000) + 1) = 0.001 / (4 + 1) = 0.0002`

## Python Code Example

```python
import math

def compute_generic_lr(dataset_size, time_factor=1.0):
    # Returns a recommended learning rate for generic models
    return 0.001 / (math.log10(dataset_size) + 1) / time_factor

# Example usage:
dataset_size = 10000
learning_rate = compute_generic_lr(dataset_size)
print(f"Recommended learning rate: {learning_rate:.6f}")  # Output: 0.000200
```

## Notes

- Adjust time_factor down for shorter training, up for longer/more conservative training.
- Use this when no literature specific rule applies.

---
