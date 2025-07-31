# hyperassist/parameter_assist/formulas/l2_param_heuristics.py 

from typing import Tuple 
import math 

def recommend_batch_size(
    ram_gb: float, 
    bytes_per_sample: int,
    buffer_factor: float = 1.5,
    compute_factor: float = 1.0,
    max_batch: int = 512
) -> Tuple[int, str, str]:
    ram_bytes = ram_gb * 1024**3 
    est_batch = ram_bytes / (bytes_per_sample * buffer_factor * compute_factor)
    batch_size = int(min(max_batch, est_batch))
    formula = "batch = min(max_batch, RAM / (sample_size * buffer * compute))"
    explanation = (
        f"Estimates batch size based on available RAM ({ram_gb}GB), memory per sample "
        f"({bytes_per_sample}B), and safety buffers. buffer_factor and compute_factor "
        f"adjust for memory spikes and hardware speed."
    )
    return batch_size, formula, explanation

def recommend_total_steps(
    dataset_size: int,
    batch_size: int,
    epochs: int
) -> Tuple[int, str, str]:
    steps = int((dataset_size / batch_size) * epochs)
    formula = "total_steps = (dataset_size / batch_size) * epochs"
    explanation = (
        f"Total training steps calculated from dataset size ({dataset_size}, batch size)"
        f"({batch_size}, and number of epochs ({epochs}).)"
    )
    return steps, formula, explanation

def recommend_warmup_steps(
    total_steps: int,
    warmup_pct: float = 0.05
) -> Tuple[int, str, str]:
    warmup = int(total_steps * warmup_pct)
    formula = "warmup_steps = total_steps * warmup_pct"
    explanation = (
        f"Warmup steps are {warmup_pct*100:.1f}% of total steps ({total_steps})."
        f"This helps stabilize training early on."
    )
    return warmup, formula, explanation

def recommend_gradient_accum_steps(
    effective_batch_size: int,
    per_device_batch: int,
    num_devices: int
) -> Tuple[int, str, str]:
    denom = per_device_batch * num_devices
    accum_steps = int(max(1, math.ceil(effective_batch_size / denom)))
    formula = "accum_steps = effective_batch / (per_device_batch * num_devices"
    explanation = (
        f"Computes accumulation steps to reach effective batch size of {effective_batch_size}, "
        f"given {per_device_batch} per device batch and {num_devices} devices "
    )
    return accum_steps, formula, explanation

def recommend_param_init_scale(
    fan_in: int,
    method: str = "xavier"
) -> Tuple[float, str, str]:
    method = method.lower()
    if method == "xavier":
        scale = 1.0 / math.sqrt(fan_in)
        formula = "scale = 1 / sqrt(fan_in)"
        explanation = f"Xavier initialization: recommended for tanh/sigmoid activations. fan_in = {fan_in}."
    elif method in ("he", "kaiming_uniform"):
        scale = math.sqrt(2.0 / fan_in)
        formula = "scale = sqrt(2 / fan_in)"
        explanation = (
            f"He (Kaiming) initialization: recommended for ReLU-type activations. "
            f"fan_in = {fan_in}."
        )
    else:
        raise ValueError(f"Unknown init method: {method}. Supported: 'xavier', 'he', 'kaiming_uniform'.")
    return scale, formula, explanation

def recommend_embedding_dim(
        vocab_size: int, 
        depth: int = 6 
) -> Tuple[int, str, str]:
    dim = int(math.log2(vocab_size) * depth)
    clamped = max(128, min(2048, dim))
    formula = "embedding_dim = clamp(log2(vocab_size) * depth, 128, 2048)"
    explanation = (
        f"Embedding dim scaled from vocab_size = {vocab_size} and model depth = {depth}. "
        f"Clamped between 128 and 2048 for stability. "
    )
    return clamped, formula, explanation
