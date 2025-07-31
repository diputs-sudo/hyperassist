# hyperassist/utils/color_utils.py

"""
Why not use a package like colorama?

Colorama is a great package but for this use case, where I want the CLI to run with minimum delay,
I prefer not to introduce a large external color package. The goal of PBX is to remain as native and 
lightweight as possible avoiding extra downloads where not needed.

Therefore, I implemented this minimal color_utils.py instead.
"""

def color(text, color_name, bold=False):
    colors = {
        "reset": "\033[0m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "purple": "\033[0;35m", 
        "dark_purple": "\033[38;5;17m"
    }
    color_code = colors.get(color_name, colors["reset"])
    if bold: 
        color_code ="\033[1m" + color_code
    return color_code + text + colors["reset"]

# Borrow it from https://github.com/diputs-sudo/pbx 