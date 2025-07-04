# Batch Size Calculation Formula

## Formula

```
batch = min(max_batch, ram_bytes / (bytes_per_sample * buffer_factor * compute_factor))
```

- `max_batch`: The upper limit for batch size (may be set by model or hardware).
- `ram_bytes`: Total RAM (or GPU VRAM) available for training, in bytes.
- `bytes_per_sample`: Memory needed per data sample (estimate or empirical).
- `buffer_factor`: Safety margin for unexpected memory spikes (e.g., 1.2 or 2.0).
- `compute_factor`: Adjusts batch size based on CPU/GPU power (e.g., 1 for “normal,” <1 for slow hardware).

## Explanation

This formula estimates the largest safe batch size for your available memory and compute:

- It divides available memory by the approximate cost per sample (with a safety buffer), so you avoid running out of memory.
- `buffer_factor` prevents out of memory (OOM) crashes by reserving some headroom.
- `compute_factor` lets you reduce batch size on slower devices to prevent overloading.

> **Example:**  
> - `ram_bytes = 8GB = 8 * 1024^3 = 8589934592`  
> - `bytes_per_sample = 2048`  
> - `buffer_factor = 1.5`  
> - `compute_factor = 1.0`  
> - `max_batch = 512`  
>
> `batch = min(512, 8589934592 / (2048 * 1.5 * 1.0)) = min(512, 2792) = 512`

## Python Code Example

```python
def compute_batch_size(ram_bytes, bytes_per_sample, buffer_factor=1.5, compute_factor=1.0, max_batch=512):
    est_batch = ram_bytes / (bytes_per_sample * buffer_factor * compute_factor)
    return int(min(max_batch, est_batch))

# Example usage:
ram_bytes = 8 * 1024**3  # 8GB RAM
bytes_per_sample = 2048
batch_size = compute_batch_size(ram_bytes, bytes_per_sample)
print(f"Recommended batch size: {batch_size}")  # Output: 512
```

## Notes
- Empirical adjustment is sometimes needed (some frameworks have extra memory overhead).
- Set max_batch based on the model, task, or practical testing.
- For most users, buffer_factor between 1.2 and 2.0 is safe.