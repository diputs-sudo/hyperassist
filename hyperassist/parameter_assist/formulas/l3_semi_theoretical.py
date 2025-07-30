# hyperassist/parameter_assist/formulas/l3_semi_theoretical.py

from typing import Tuple
import math

def recommend_transformer_learning_rate(
    step: int, 
    d_model: int = 512, 
    warmup_steps: int = 4000
) -> Tuple[float, str, str]:
    scale = d_model ** -0.5
    arg1 = step ** -0.5
    arg2 = step / (warmup_steps ** 1.5)
    lr = scale * min(arg1, arg2)
    formula = "lr = d_model^-0.5 * min(step^-0.5, step / warmup_steps^1.5)"
    explanation = (
        f"Transformer learning rate schedule from 'Attention is All You Need': "
        f"scales as d_model^-0.5 and uses warmup ({warmup_steps})."
    )
    return lr, formula, explanation

def recommend_dropout_rate(
    model_complexity: float
) -> Tuple[float, str, str]:
    rate = min(0.6, max(0.1, 0.25 + 1/(model_complexity + 2)))
    formula = "dropout = clamp(0.25 + 1/(complexity+2), 0.1, 0.6)"
    explanation = (
        f"Sets dropout inversely to model complexity ({model_complexity}), "
        "clamped to [0.1, 0.6]. Higher complexity → lower dropout."
    )
    return rate, formula, explanation

def recommend_hidden_size(
    parameter_budget: int,
    num_layers: int,
    multiple_of: int = 64
) -> Tuple[int, str, str]:
    # Approximate hidden size so that hidden_size^2 * num_layers ≈ parameter_budget
    h_est = int(math.sqrt(parameter_budget / num_layers))
    # Round to nearest multiple (64 is common for GPUs)
    hidden_size = int(round(h_est / multiple_of) * multiple_of)
    formula = "hidden_size ≈ round(sqrt(parameter_budget / num_layers), multiple_of)"
    explanation = (
        f"Estimates hidden size so that hidden_size^2 * num_layers ≈ parameter budget "
        f"({parameter_budget}). Rounded to nearest {multiple_of}."
    )
    return hidden_size, formula, explanation

def recommend_num_heads(
    hidden_size: int
) -> Tuple[int, str, str]:
    # Heads should divide hidden_size exactly, with typical range [4, 32]
    for h in range(8, 33):
        if hidden_size % h == 0:
            num_heads = h
            break
    else:
        num_heads = 8  # fallback
    formula = "num_heads divides hidden_size evenly, usually in [8, 32]"
    explanation = (
        f"Selects number of attention heads that evenly divides hidden_size ({hidden_size}), "
        "preferably between 8 and 32."
    )
    return num_heads, formula, explanation

def recommend_ffn_size(
    hidden_size: int
) -> Tuple[int, str, str]:
    ffn = 4 * hidden_size
    formula = "ffn_size = 4 * hidden_size"
    explanation = (
        "Feedforward network (FFN) size is typically 4× hidden_size, "
        "as in Transformer and BERT models."
    )
    return ffn, formula, explanation

def recommend_label_smoothing(
    model_type: str
) -> Tuple[float, str, str]:
    smoothing = 0.1 if model_type.lower() in {"transformer", "t5", "bart", "gpt"} else 0.0
    formula = "label_smoothing = 0.1 for transformers, else 0.0"
    explanation = (
        f"Sets label smoothing to 0.1 for Transformer-like models ({model_type}), "
        "0.0 otherwise. Matches BERT/T5/BART standards."
    )
    return smoothing, formula, explanation

def recommend_attention_dropout(
    model_type: str, risk_level: str = "normal"
) -> Tuple[float, str, str]:
    if model_type.lower() in {"transformer", "t5", "gpt", "bert"}:
        rate = 0.1 if risk_level == "normal" else 0.15
    else:
        rate = 0.0
    formula = "attention_dropout = 0.1 (transformers), else 0.0"
    explanation = (
        f"Sets attention dropout for {model_type}. Default 0.1 for Transformers, "
        f"increased to 0.15 for high-overfitting risk."
    )
    return rate, formula, explanation
