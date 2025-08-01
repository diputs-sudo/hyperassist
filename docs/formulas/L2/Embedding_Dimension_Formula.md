# Embedding Dimension Formula

## Formula

```
embedding_dim = clamp(log2(vocab_size) * depth, 128, 2048)
```
- `embedding_dim`: The recommended embedding dimension.
- `vocab_size`: The size of the vocabulary.
- `depth`: The model’s depth (e.g., number of layers), default is 6.
- `clamp`: Ensures the value stays between 128 and 2048.

## Explanation

This formula estimates a reasonable embedding dimension for a model based on vocabulary size and model depth.  
Take the base-2 logarithm of the vocabulary size and multiply by the number of layers (depth). Clamp the result to a minimum of 128 and a maximum of 2048 for stability and memory efficiency.

> **Example:**  
> - `vocab_size = 30,000`  
> - `depth = 6`  
> - `embedding_dim = clamp(log2(30000) * 6, 128, 2048) ≈ clamp(14.87 * 6, 128, 2048) ≈ clamp(89.2, 128, 2048) = 128`

## Python Code Example

```python
import math

def recommend_embedding_dim(
    vocab_size: int, 
    depth: int = 6 
) -> int:
    dim = int(math.log2(vocab_size) * depth)
    clamped = max(128, min(2048, dim))
    return clamped

# Example usage:
embedding_dim = recommend_embedding_dim(30000, 6)
print(f"Recommended embedding dimension: {embedding_dim}")  # Output: 128
```

## Notes
- Increasing depth leads to larger embedding dimensions.
- Embedding sizes below 128 or above 2048 may be unstable or wasteful.
- For very large models or vocabularies, empirical tuning is recommended.

---