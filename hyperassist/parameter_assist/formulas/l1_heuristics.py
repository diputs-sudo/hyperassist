# hyperassist/parameter_assist/formulas/l1_heuristics.py

from typing import Tuple

def recommend_cnn_learning_rate(batch_size: int) -> Tuple[float, str, str]:
    """
    Empirical LR scaling rule for CNNs from Goyal et al. (2017)
    """
    lr = 0.1 * (batch_size / 256)
    formula = "lr = 0.1 * (batch_size / 256)"
    explanation = (
        "Uses the linear scaling rule from Goyal et al. (2017), where learning rate scales "
        "proportionally with batch size. Baseline: 0.1 at batch size 256."
    )
    return lr, formula, explanation


def recommend_weight_decay(compute_factor: float = 1.0) -> Tuple[float, str, str]:
    """
    Scales weight decay with compute factor. Common base is 1e-5.
    """
    decay = 1e-5 * compute_factor
    formula = "weight_decay = 1e-5 * compute_factor"
    explanation = (
        "Weight decay is scaled based on available compute. 1e-5 is a common base for AdamW. "
        "Increase compute_factor (e.g. 2.0) for heavier regularization; decrease it on low-end hardware."
    )
    return decay, formula, explanation


def recommend_optimizer(model_type: str) -> Tuple[str, str, str]:
    """
    Chooses default optimizer by model type.
    """
    if model_type.lower() in {"transformer", "bert", "gpt"}:
        opt = "AdamW"
    elif model_type.lower() in {"cnn", "resnet"}:
        opt = "SGD"
    else:
        opt = "Adam"
    formula = f"optimizer = {opt} (based on model_type)"
    explanation = f"Selects optimizer empirically based on model_type: {model_type}."
    return opt, formula, explanation


def recommend_activation(model_type: str) -> Tuple[str, str, str]:
    """
    Chooses activation function based on model type.
    """
    model = model_type.lower()
    if model in {"transformer", "bert", "gpt"}:
        act = "GELU"
    elif model in {"modern_cnn", "convnext"}:
        act = "SiLU"
    else:
        act = "ReLU"
    formula = f"activation = {act} (based on model_type)"
    explanation = f"Uses common activation heuristic: {act} for {model_type} models."
    return act, formula, explanation


def recommend_scheduler_type(total_epochs: int) -> Tuple[str, str, str]:
    """
    Select scheduler type based on number of epochs.
    """
    if total_epochs >= 20:
        sched = "cosine"
    else:
        sched = "linear"
    formula = "scheduler = 'cosine' if epochs ≥ 20 else 'linear'"
    explanation = (
        "Chooses scheduler type based on training length. Cosine is preferred for longer training, "
        "linear warmup/decay for shorter runs."
    )
    return sched, formula, explanation


def recommend_dataloader_workers(cpu_cores: int, num_gpus: int) -> Tuple[int, str, str]:
    """
    Recommend number of dataloader workers.
    """
    workers = min(cpu_cores, 4 * num_gpus)
    formula = "workers = min(cpu_cores, 4 * num_gpus)"
    explanation = (
        "Sets number of data loader workers based on available CPU and GPU parallelism. "
        "Assumes a good balance between CPU prefetching and GPU consumption."
    )
    return workers, formula, explanation
