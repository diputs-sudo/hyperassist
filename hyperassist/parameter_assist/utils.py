# hyperassist/parameter_assist/utils.py

import os
import json
import csv 

CANONICAL_PARAM_MAP = {
    ("lr", "learningrate", "learningRate", "eta", "init_lr", "base_lr", "final_lr"): "learning_rate",
    ("per_device_train_batch_size", "train_batch_size", "batchSize", "mini_batch_size"): "batch_size",
    ("dropout_rate", "keep_prob", "p_dropout"): "dropout",
    ("optim", "optim_type", "optimizer_type"): "optimizer",
    ("weightDecay", "l2"): "weight_decay",
    ("num_workers", "n_jobs", "workers"): "cpu_cores",
    ("num_epochs", "epochs", "n_epochs"): "total_epochs",
    ("n_layers", "layers"): "num_layers",
    ("hidden", "hidden_dim", "hidden_dims"): "hidden_size",
    ("embedding_dim", "emb_dim"): "embedding_size",
    ("vocab", "vocab_size"): "vocab_size",
    ("train_steps", "steps"): "total_steps",
}

def canonicalize_params(params):
    alias_to_canonical = {}
    for aliases, canonical in CANONICAL_PARAM_MAP.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical

    return {alias_to_canonical.get(k.lower(), k): v for k, v in params.items()}

def detect_cpu_cores():
    import psutil
    try:
        return psutil.cpu_count()
    except Exception:
        import os
        return os.cpu_count() or 1

def detect_ram():
    import psutil
    try:
        return psutil.virtual_memory().total // (1024 ** 3)
    except Exception:
        return 8

def detect_dataset_size(datasets_file=None, datasets_folder=None):
    """Estimate dataset size (number of samples) from file/folder."""
    if datasets_folder and os.path.isdir(datasets_folder):
        count = 0
        for root, dirs, files in os.walk(datasets_folder):
            files = [f for f in files if not f.startswith('.')]
            count += len(files)
        return count
    elif datasets_file and os.path.isfile(datasets_file):
        ext = os.path.splitext(datasets_file)[-1].lower()
        if ext == ".json":
            with open(datasets_file, "r") as f:
                try:
                    data = json.load(f)
                    return len(data)
                except Exception:
                    f.seek(0)
                    return sum(1 for _ in f)
        elif ext in [".jsonl", ".txt"]:
            with open(datasets_file) as f:
                return sum(1 for _ in f)
        elif ext in [".csv", ".tsv"]:
            with open(datasets_file, newline='') as f:
                reader = csv.reader(f)
                return sum(1 for _ in reader) - 1  
        with open(datasets_file) as f:
            return sum(1 for _ in f)
    return None

def build_auto_context(user_context):
    """Returns a dict merging user input with auto-detected system info."""
    context = dict(user_context)
    if "cpu_cores" not in context:
        context["cpu_cores"] = detect_cpu_cores()
    if "ram_gb" not in context:
        context["ram_gb"] = detect_ram()
    if "datasets_file" in context or "datasets_folder" in context:
        size = detect_dataset_size(
            context.get("datasets_file"),
            context.get("datasets_folder")
        )
        if size is not None:
            context["dataset_size"] = size
    return context
