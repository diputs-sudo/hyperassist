# hyperassist/parameter_assist/suggestor.py

from .formula_registry import discover_formulas
from .utils import canonicalize_params, detect_ram
from ..utils.color_utils import color

def check(params, **kwargs):
    """
    Main API: Analyze hyperparameters and print best-practice advice.
    Automatically discovers and runs all compatible formulas.
    """
    params = canonicalize_params(params)
    context = {**params, **kwargs}
    if 'ram_gb' not in context:
        context['ram_gb'] = detect_ram()
    theory_on = (kwargs.get('theory', '').lower() == 'on')
    formulas = discover_formulas(include_theoretical=theory_on)

    ran_any = False
    missing = []

    print(color("\n[HyperAssist] Parameter Recommendation Report\n", "yellow"))

    for f in formulas:
        needed_args = f["args"]
        if all(arg in context for arg in needed_args):
            ran_any = True
            try:
                results = f["function"](**{k: context[k] for k in needed_args})
            except Exception as e:
                print(f"[{f['func_name']}]")
                print(color(f"  [!] Error running formula: {e}\n", "red"))
                continue
            value, formula, explanation = results
            print(color(f"[{f['func_name']}]", "cyan", bold=True))
            print("           ")
            print(color("  > Suggest: ", "green", bold=True) + color(f"{value}", "yellow"))
            print("           ")
            print(color("  > Formula: ", "green", bold=True) + color(f"{formula}", "white"))
            print("           ")
            print(color("  > Why:     ", "green", bold=True) + color(f"{explanation}", "white"))
            print("           ")
            print("           ")
            # Don't ask me what my partner is doing ... (interns...)
        else:
            missing.append((f["func_name"], [arg for arg in needed_args if arg not in context]))

    if missing:
        print(color("[!] Skipped formulas due to missing arguments:", "red"))
        for func_name, missing_args in missing:
            print(color(f"    - {func_name}: missing {missing_args}", "yellow"))
    if not ran_any:
        print(color("[!] No formulas could be run with the provided parameters and context.", "red"))
