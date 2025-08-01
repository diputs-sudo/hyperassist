# Total Training Steps Formula

## Formula

```
total_steps = (dataset_size / batch_size) * epochs
```
- `total_steps`: The total number of training steps.
- `dataset_size`: Number of samples in the training dataset.
- `batch_size`: Number of samples per batch.
- `epochs`: Number of passes through the full dataset.

## Explanation

This formula calculates the total number of training steps required for your experiment.  
It divides the total dataset size by the batch size to get the number of steps per epoch, then multiplies by the total number of epochs.

> **Example:**  
> - `dataset_size = 50_000`  
> - `batch_size = 100`  
> - `epochs = 10`  
> - `total_steps = (50_000 / 100) * 10 = 5,000`

## Python Code Example

```python
def recommend_total_steps(
    dataset_size: int,
    batch_size: int,
    epochs: int
) -> int:
    steps = int((dataset_size / batch_size) * epochs)
    return steps

# Example usage:
steps = recommend_total_steps(50000, 100, 10)
print(f"Recommended total steps: {steps}")  # Output: 5000
```
## Notes
- Adjust for any data augmentation or subsampling that change effective dataset size.
- Useful for setting scheduler and warmup parameters.
- For distributed training, use global batch size.

---
