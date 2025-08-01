# Dataloader Workers Formula

## Formula

```
workers = min(cpu_cores, 4 * num_gpus)
```
- `workers`: The recommended number of dataloader worker processes.
- `cpu_cores`: The number of available CPU cores.
- `num_gpus`: The number of available GPUs.

## Explanation

This formula helps set the number of parallel dataloader worker processes for efficient data loading.

Set the number of workers to the smaller of the total CPU cores or four times the number of GPUs. This balances CPU prefetching with the capacity of your GPUs, ensuring efficient data throughput without oversubscribing your system.

> **Example:**  
> - `cpu_cores = 8`  
> - `num_gpus = 2`  
> - `workers = min(8, 4 * 2) = 8`

## Python Code Example

```python
def recommend_dataloader_workers(cpu_cores: int, num_gpus: int) -> int:
    """
    Recommend number of dataloader workers.
    """
    return min(cpu_cores, 4 * num_gpus)

# Example usage:
workers = recommend_dataloader_workers(12, 3)
print(f"Recommended dataloader workers: {workers}")  # Output: 12
```

## Notes
- Increasing workers can improve throughput but may use more system RAM.
- If you see CPU bottlenecks or data loading lag, try raising the number of workers.
- For I/O heavy datasets or limited RAM, start with fewer workers and test scaling.
- Some deep learning frameworks may limit or manage workers differently.

---