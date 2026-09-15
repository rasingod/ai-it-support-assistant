"""Shared validation used at both workflow and storage boundaries."""
import re

CATEGORIES = ("VPN", "Laptop", "Email", "Printer", "Software", "Wi-Fi", "Network", "Hardware", "Other")


def canonical_category(value):
    if not isinstance(value, str):
        return None
    aliases = {c.casefold(): c for c in CATEGORIES}
    aliases.update({"wifi": "Wi-Fi", "wi fi": "Wi-Fi", "printing": "Printer"})
    return aliases.get(value.strip().casefold())


def category_answer(value):
    """Accept a category alone or Category. Description without storing prose as a category."""
    parts = re.split(r"[.:;]\s*", value.strip(), maxsplit=1)
    category = canonical_category(parts[0])
    return category, parts[1].strip() if category and len(parts) == 2 else None
