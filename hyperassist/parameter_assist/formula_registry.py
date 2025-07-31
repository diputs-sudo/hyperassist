# hyperassist/parameter_assist/formula_registry.py

import inspect
import importlib
import pkgutil
from pathlib import Path

def discover_formulas(include_theoretical=False):
    formulas = []
    formulas_dir = Path(__file__).parent / "formulas"

    for finder, name, ispkg in pkgutil.iter_modules([str(formulas_dir)]):
        if not name.startswith('l'):
            continue  
        if name == "l4_theoretical" and not include_theoretical:
            continue
        module = importlib.import_module(f".formulas.{name}", package="hyperassist.parameter_assist")
        level = name.upper().split('_')[0]  
        for func_name, func in inspect.getmembers(module, inspect.isfunction):
            if func_name.startswith(("recommend_", "estimate_")):
                sig = inspect.signature(func)
                arg_names = list(sig.parameters.keys())
                formulas.append({
                    "function": func,
                    "args": arg_names,
                    "func_name": func_name,
                    "level": level,
                    "module": name,
                })
    return formulas
