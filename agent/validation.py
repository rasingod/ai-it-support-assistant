"""Shared validation used at both workflow and storage boundaries."""
import re

CATEGORIES = ("VPN", "Laptop", "Email", "Printer", "Software", "Internet", "Mobile", "Account Access", "Wi-Fi", "Network", "Hardware", "Other")
ALIASES = {
    "wifi": "Wi-Fi", "wi fi": "Wi-Fi", "printing": "Printer",
    "phone": "Mobile", "cell phone": "Mobile", "cellphone": "Mobile",
    "pc": "Laptop", "computer": "Laptop", "desktop": "Laptop", "notebook": "Laptop",
    "outlook": "Email", "mail": "Email", "e-mail": "Email",
    "app": "Software", "application": "Software", "program": "Software",
    "mfa": "Account Access", "2fa": "Account Access", "login": "Account Access",
    "password": "Account Access",
}


def canonical_category(value):
    if not isinstance(value, str):
        return None
    aliases = {c.casefold(): c for c in CATEGORIES}
    aliases.update(ALIASES)
    return aliases.get(value.strip().casefold())


def category_answer(value):
    """Extract known whole-word categories/aliases, preserving remaining description."""
    if not isinstance(value, str) or not value.strip():
        return None, None
    exact = canonical_category(value)
    if exact:
        return exact, None
    names = {c.casefold(): c for c in CATEGORIES}
    names.update(ALIASES)
    pattern = r"(?<!\w)(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?!\w)"
    match = re.search(pattern, value, re.I)
    if not match:
        return None, None
    leftover = (value[:match.start()] + value[match.end():]).lstrip(" ,.-:;").rstrip()
    return names[match.group().casefold()], leftover or None
