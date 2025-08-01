# Warmup Steps Formula

## Formula

```
warmup_steps = total_steps * warmup_pct
```
- `warmup_steps`: The number of steps to linearly increase the learning rate at the start of training.
- `total_steps`: The total number of training steps.
- `warmup_pct`: Fraction of training steps used for warmup (default is 0.05, or 5%).

## Explanation

This formula determines how many steps should be used for learning rate warmup at the start of training.  
A short warmup period helps stabilize early training, especially for large models or with high learning rates. The percentage is typically set between 3% and 10% of the total training steps.

> **Example:**  
> - `total_steps = 10,000`  
> - `warmup_pct = 0.05`  
> - `warmup_steps = 10,000 * 0.05 = 500`

## Python Code Example

```python
def recommend_warmup_steps(
    total_steps: int,
    warmup_pct: float = 0.05
) -> int:
    warmup = int(total_steps * warmup_pct)
    return warmup

# Example usage:
warmup = recommend_warmup_steps(10000)
print(f"Recommended warmup steps: {warmup}")  # Output: 500
```

## Notes
- If training is unstable early on, try increasing the warmup percentage.
- Some schedules use a fixed number of warmup steps instead of a percentage.
- Warmup is especially important for transformer models and large-batch training.

---