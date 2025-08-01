# Parameter Initialization Scale Formula

## Formula

### Xavier Initialization

```
scale = 1 / sqrt(fan_in)
```

### He Initialization

```
scale = sqrt(2 / fan_in)
```

- `scale`: Scaling factor for weight initialization.
- `fan_in`: The number of input units to the layer.
- `method`: Initialization method, usually "xavier" or "he".

## Explanation

This formula sets the scale for initializing neural network weights, based on the chosen initialization method and the number of input connections (`fan_in`).

For Xavier (Glorot) initialization, use `1 / sqrt(fan_in)`.  
For He (Kaiming) initialization, use `sqrt(2 / fan_in)`.  
Xavier works well for tanh or sigmoid activations, while He is designed for ReLU-like activations.

> **Example (Xavier):**  
> - `fan_in = 256`  
> - `scale = 1 / sqrt(256) = 0.0625`

> **Example (He):**  
> - `fan_in = 256`  
> - `scale = sqrt(2 / 256) ≈ 0.0884`

## Python Code Example

```python
import math

def recommend_param_init_scale(
    fan_in: int,
    method: str = "xavier"
) -> float:
    method = method.lower()
    if method == "xavier":
        return 1.0 / math.sqrt(fan_in)
    elif method == "he":
        return math.sqrt(2.0 / fan_in)
    else:
        raise ValueError(f"Unknown init method: {method}")

# Example usage:
scale = recommend_param_init_scale(256, "he")
print(f"Recommended init scale: {scale}")  # Output: 0.088388...
```

## Notes
- Xavier is standard for layers with tanh or sigmoid activations.
- He initialization is standard for layers with ReLU or variants.
- Always match the initialization method to your activation function for stable training.

---