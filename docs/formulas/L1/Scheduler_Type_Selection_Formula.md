# Scheduler Type Selection Formula

## Formula

```
scheduler = 'cosine' if epochs ≥ 20 else 'linear'
```
- `scheduler`: The recommended learning rate scheduler type.
- `total_epochs`: The total number of training epochs.

## Explanation

This guideline selects the scheduler type based on the planned training duration.

If you train for 20 or more epochs, use a cosine scheduler. Cosine annealing is effective for longer runs and helps gradually decrease the learning rate toward the end of training.  
If you train for fewer than 20 epochs, use a linear scheduler. Linear schedules are simple and effective for short training runs.

> **Example:**  
> - `total_epochs = 50`  
> - `scheduler = 'cosine'`

## Python Code Example

```python
def recommend_scheduler_type(total_epochs: int) -> str:
    """
    Select scheduler type based on number of epochs.
    """
    if total_epochs >= 20:
        return "cosine"
    else:
        return "linear"

# Example usage:
scheduler = recommend_scheduler_type(12)
print(f"Recommended scheduler: {scheduler}")  # Output: linear
```
## Notes
- Cosine schedulers often work well with longer warmup periods and gradual decay.
- For very short runs or fast experiments, a linear schedule can save tuning time.
- Advanced schedulers (like polynomial or step) can be used for specific needs.

---