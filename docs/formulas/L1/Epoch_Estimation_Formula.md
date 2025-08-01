# Epoch Count Estimation Formula

## Formula
```
epoch_count = total_steps / (dataset_size / batch_size)
```

- `epoch_count`: The estimated number of full training epochs (rounded to nearest integer).
- `total_steps`: The total number of training steps (iterations).
- `dataset_size`: The total number of training samples.
- `batch_size`: The number of samples per batch.

## Explanation

This formula estimates how many full passes (epochs) through the dataset your training run will complete, based on the total number of training steps and batch size. Since you can’t run a fractional epoch in most practical settings, the result is rounded to the nearest integer.

It works by calculating how many steps are required to process the entire dataset once, then divides the total number of steps by that value and rounds the result.

> **Example:**  
> - `total_steps = 5000`  
> - `dataset_size = 100000`  
> - `batch_size = 64`  
> - `epoch_count = round(5000 / (100000 / 64)) = round(3.2) = 3`

## Python Code Example

```python
def estimate_epoch_count(total_steps: int, dataset_size: int, batch_size: int) -> int:
    """
    Estimate number of epochs based on total training steps, dataset size, and batch size.
    Rounded to the nearest integer.
    """
    return round(total_steps / (dataset_size / batch_size))

# Example usage:
epochs = estimate_epoch_count(total_steps=5000, dataset_size=100000, batch_size=64)
print(f"Estimated epoch count: {epochs}")  # Output: 3
```

## Notes
- Useful when tuning schedule-based hyperparameters like learning rate decay or warmup length.
- If you're given `total_steps` by a scheduler or optimizer config, this helps you relate it to actual training duration.
- You can invert this to calculate `total_steps = epoch_count * (dataset_size / batch_size)` if needed.

---