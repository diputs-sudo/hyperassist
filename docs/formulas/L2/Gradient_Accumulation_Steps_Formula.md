# Gradient Accumulation Steps Formula

## Formula

```
accum_steps = effective_batch / (per_device_batch * num_devices)
```
- `accum_steps`: The number of gradient accumulation steps.
- `effective_batch`: The desired total batch size across all devices.
- `per_device_batch`: The batch size processed by each device per step.
- `num_devices`: The number of devices (e.g., GPUs or TPUs) used for training.

## Explanation

This formula calculates how many steps to accumulate gradients before performing an optimizer update, allowing you to achieve a larger effective batch size than what fits into device memory.  
Divide the target effective batch size by the product of the per device batch size and the number of devices.  
Always round up to ensure the effective batch is met or exceeded.

> **Example:**  
> - `effective_batch = 1024`  
> - `per_device_batch = 128`  
> - `num_devices = 4`  
> - `accum_steps = 1024 / (128 * 4) = 2`

## Python Code Example

```python
import math

def recommend_gradient_accum_steps(
    effective_batch_size: int,
    per_device_batch: int,
    num_devices: int
) -> int:
    denom = per_device_batch * num_devices
    accum_steps = int(max(1, math.ceil(effective_batch_size / denom)))
    return accum_steps

# Example usage:
accum = recommend_gradient_accum_steps(1024, 128, 4)
print(f"Recommended gradient accumulation steps: {accum}")  # Output: 2
```

## Notes
- Accumulation lets you train with very large effective batch sizes even on small GPUs.
- Always use `ceil` to avoid underestimating steps.
- Watch for increased memory usage during accumulation.
- Some frameworks handle accumulation automatically, others require manual setup.

---