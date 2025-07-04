import math
import numpy as np

def recommend_lr(params):
    """Return (recommended_lr, formula, explanation) based on model type and batch size."""
    model_type = params.get("model_type", "generic").lower()
    batch_size = params.get("per_device_train_batch_size", 32)
    d_model = params.get("hidden_size", 512)
    step = params.get("step", 1000)
    warmup = params.get("warmup_steps", 4000)
    dataset_size = params.get("dataset_size", 10000)
    time_factor = params.get("time_factor", 1.0)

    if model_type == "cnn":
        lr = 0.1 * (batch_size / 256)
        formula = "lr = 0.1 * (batch_size / 256)"
        expl = "Linear scaling rule for CNNs (Goyal et al., 2017)"
    elif model_type == "transformer":
        lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
        formula = "lr = d_model^(-0.5) * min(step^(-0.5), step / warmup_steps^1.5)"
        expl = "Transformer schedule (Vaswani et al., 2017)"
    else:
        lr = 0.001 / (math.log10(dataset_size) + 1) / time_factor
        formula = "lr = 0.001 / (log10(dataset_size)+1) / time_factor"
        expl = "Generic rule based on dataset size and time preference"
    return lr, formula, expl

def recommend_batch_size(params):
    ram_gb = float(params.get("ram_gb", 8))
    input_dim = int(params.get("input_dim", 512))
    buffer_factor = 1.5
    compute_factor = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(params.get("compute", "medium").lower(), 1.0)
    max_batch = {"low": 64, "medium": 128, "high": 256}.get(params.get("compute", "medium").lower(), 128)
    total_ram_bytes = ram_gb * (1024 ** 3)
    bytes_per_sample = input_dim * 4
    est = total_ram_bytes / (bytes_per_sample * buffer_factor * compute_factor)
    batch = int(min(max_batch, max(1, est)))
    formula = "batch = min(max_batch, ram_bytes/(bytes_per_sample * buffer_factor * compute_factor))"
    expl = "RAM-driven batch size estimation"
    return batch, formula, expl

def recommend_dropout(params):
    complexity = float(params.get("data_complexity", 6))  # log2(input_dim+1) often
    raw = 0.25 + 1/(complexity + 2)
    dropout = min(0.6, max(0.1, raw))
    formula = "dropout = clamp(0.25 + 1/(complexity+2),0.1,0.6)"
    expl = "Complexity-based dropout suggestion"
    return dropout, formula, expl

def recommend_weight_decay(params):
    compute_factor = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(params.get("compute", "medium").lower(), 1.0)
    wd = 1e-5 * compute_factor
    formula = "weight_decay = 1e-5 * compute_factor"
    expl = "Compute factor scaled weight decay"
    return wd, formula, expl

def check(args):
    """
    Formula-driven smart check of hyperparameters for typical deep learning setups.
    Compares to literature-backed recommendations and prints explanation.
    """
    params = args if isinstance(args, dict) else vars(args)
    params = params.copy()

    input_shape = params.get("input_shape", "512")
    if isinstance(input_shape, (tuple, list)):
        input_dim = math.prod(input_shape)
    else:
        dims = [int(x) for x in str(input_shape).replace(',', 'x').replace(' ', 'x').split('x') if x.isdigit()]
        input_dim = math.prod(dims) if dims else 512
    params["input_dim"] = input_dim

    params.setdefault("ram_gb", 8)
    params.setdefault("compute", "medium")
    params.setdefault("data_complexity", math.log2(input_dim + 1))

    print("\n[parameter_assist] Formula based checks and suggestions:")

    lr = float(params.get("learning_rate", 1e-3))
    rec_lr, lr_formula, lr_expl = recommend_lr(params)
    if lr > 0 and rec_lr > 0 and (lr/rec_lr > 2 or lr/rec_lr < 0.5):
        print(f"  [!] learning_rate={lr} | Recommended={rec_lr:.2g}")
        print(f"      Formula: {lr_formula}")
        print(f"      Reason: {lr_expl}")
    else:
        print(f"  [OK] learning_rate={lr} ~ {rec_lr:.2g} ({lr_expl})")

    bsz = int(params.get("per_device_train_batch_size", 32))
    rec_bsz, bsz_formula, bsz_expl = recommend_batch_size(params)
    if bsz/rec_bsz > 2 or bsz/rec_bsz < 0.5:
        print(f"  [!] per_device_train_batch_size={bsz} | Recommended={rec_bsz}")
        print(f"      Formula: {bsz_formula}")
        print(f"      Reason: {bsz_expl}")
    else:
        print(f"  [OK] per_device_train_batch_size={bsz} ~ {rec_bsz} ({bsz_expl})")

    dropout = float(params.get("dropout", 0.2))
    rec_do, do_formula, do_expl = recommend_dropout(params)
    if abs(dropout - rec_do) > 0.15:
        print(f"  [!] dropout={dropout} | Recommended={rec_do:.2g}")
        print(f"      Formula: {do_formula}")
        print(f"      Reason: {do_expl}")
    else:
        print(f"  [OK] dropout={dropout} ~ {rec_do:.2g} ({do_expl})")

    wd = float(params.get("weight_decay", 1e-5))
    rec_wd, wd_formula, wd_expl = recommend_weight_decay(params)
    if wd/rec_wd > 2 or wd/rec_wd < 0.5:
        print(f"  [!] weight_decay={wd} | Recommended={rec_wd:.1g}")
        print(f"      Formula: {wd_formula}")
        print(f"      Reason: {wd_expl}")
    else:
        print(f"  [OK] weight_decay={wd} ~ {rec_wd:.1g} ({wd_expl})")

    print("")
