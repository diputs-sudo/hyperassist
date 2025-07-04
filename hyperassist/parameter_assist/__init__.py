from .suggestor import check

import math
import os
import json
import psutil

def _get_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        return 8 

def _count_dataset_size(datasets_file=None, datasets_folder=None):
    if datasets_file:
        if datasets_file.endswith('.json'):
            with open(datasets_file, 'r') as f:
                first = f.readline().strip()
                if first.startswith('['):  
                    data = json.load(open(datasets_file))
                    return len(data)
                else:  
                    return sum(1 for _ in open(datasets_file))
        elif datasets_file.endswith('.csv') or datasets_file.endswith('.tsv'):
            return sum(1 for _ in open(datasets_file)) - 1  
        else:
            return sum(1 for _ in open(datasets_file))
    elif datasets_folder:
        return sum([len(files) for _, _, files in os.walk(datasets_folder)])
    else:
        return None

def check(
    args,
    model_type=None,
    compute="medium",
    datasets_file=None,
    datasets_folder=None,
    dataset_size=None,
    input_shape=None
):
    params = args if isinstance(args, dict) else vars(args)
    params = params.copy()
    params["model_type"] = model_type or params.get("model_type", "generic")
    params["compute"] = compute or params.get("compute", "medium")
    params["ram_gb"] = _get_ram_gb()
    params["dataset_size"] = dataset_size or _count_dataset_size(datasets_file, datasets_folder) or 10000

    if input_shape:
        params["input_shape"] = input_shape
    else:
        params["input_shape"] = params.get("input_shape", "512")

    input_shape = params["input_shape"]
    dims = [int(x) for x in str(input_shape).replace(',', 'x').replace(' ', 'x').split('x') if x.isdigit()]
    input_dim = math.prod(dims) if dims else 512
    params["input_dim"] = input_dim
    params["data_complexity"] = math.log2(input_dim + 1)

    from .suggestor import check as formula_check
    print("[parameter_assist] Using:")
    print(f"  Model type: {params['model_type']}")
    print(f"  Compute: {params['compute']}")
    print(f"  RAM: {params['ram_gb']} GB")
    print(f"  Dataset size: {params['dataset_size']}")
    print(f"  Input shape: {params['input_shape']} (parsed as dim={input_dim})")
    formula_check(params)

__all__ = ["check"]
