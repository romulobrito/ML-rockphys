#!/usr/bin/env python3
"""
Replace Portuguese LaTeX accent escapes with UTF-8 in relatorio_porosidade_ml.tex.
Uses correct regex for TeX \\^ \\' \\~ \\c constructs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def convert_portuguese_accents(text: str) -> str:
    """Apply replacements; order matters for overlapping patterns."""
    # Tie + dotless i (Daí)
    text = text.replace(r"Da\'\i~", "Daí")
    text = text.replace(r"Da\'{\i}", "Daí")
    # domínio
    text = text.replace(r"dom\'{\i}nio", "domínio")
    # Braced i acute
    text = text.replace(r"\'{\i}", "í")
    # Dotless i acute (common in babel Portuguese)
    text = text.replace(r"\'\i", "í")

    # Acute vowels (after \'{\i} and \'\i)
    text = re.sub(r"\\'([aeiouAEIOU])", lambda m: _acute(m.group(1)), text)

    # Grave
    text = re.sub(r"\\`([aeiouAEIOU])", lambda m: _grave(m.group(1)), text)

    # Circumflex on vowels: TeX is backslash + caret + letter
    text = re.sub(r"\\\^([aeiouAEIOU])", lambda m: _circ(m.group(1)), text)

    # Tilde on a, o, n (Portuguese / Spanish)
    text = re.sub(r"\\~a", "ã", text)
    text = re.sub(r"\\~o", "õ", text)
    text = re.sub(r"\\~e", "ẽ", text)
    text = re.sub(r"\\~n", "ñ", text)
    text = re.sub(r"\\~A", "Ã", text)
    text = re.sub(r"\\~O", "Õ", text)

    # Cedilla
    text = re.sub(r"\\c\{c\}", "ç", text)
    text = re.sub(r"\\c\{C\}", "Ç", text)

    return text


def fix_spurious_accent_spaces(text: str) -> str:
    """
    Merge words broken by a space after acute-i (or similar) from old TeX \\'\\i
    conversions, e.g. anal\\'\\i tico -> analí tico. Idempotent on clean UTF-8.
    """
    pairs = [
        ("crí tica", "crítica"),
        ("analí tico", "analítico"),
        ("fí sicas", "físicas"),
        ("fí sica", "física"),
        ("fí sico", "físico"),
        ("empí ricas", "empíricas"),
        ("empí rica", "empírica"),
        ("empí ricos", "empíricos"),
        ("explí cito", "explícito"),
        ("estatí sticos", "estatísticos"),
        ("estatí sticas", "estatísticas"),
        ("estatí stica", "estatística"),
        ("contí nuo", "contínuo"),
        ("distribuí das", "distribuídas"),
        ("disponí veis", "disponíveis"),
        ("í ndices", "índices"),
        ("determiní stica", "determinística"),
        ("determiní stico", "determinístico"),
        ("séxta", "sexta"),
        ("reconstruí vel", "reconstruível"),
        ("construí do", "construído"),
        ("ruí do", "ruído"),
    ]
    for wrong, right in pairs:
        text = text.replace(wrong, right)
    text = text.replace("Daí  ", "Daí ")
    return text


def _acute(c: str) -> str:
    return {
        "a": "á",
        "e": "é",
        "i": "í",
        "o": "ó",
        "u": "ú",
        "A": "Á",
        "E": "É",
        "I": "Í",
        "O": "Ó",
        "U": "Ú",
    }[c]


def _grave(c: str) -> str:
    return {
        "a": "à",
        "e": "è",
        "i": "ì",
        "o": "ò",
        "u": "ù",
        "A": "À",
    }[c]


def _circ(c: str) -> str:
    return {
        "a": "â",
        "e": "ê",
        "i": "î",
        "o": "ô",
        "u": "û",
        "A": "Â",
        "E": "Ê",
        "I": "Î",
        "O": "Ô",
        "U": "Û",
    }[c]


def main() -> int:
    path = Path(__file__).resolve().parent / "relatorio_porosidade_ml.tex"
    raw = path.read_text(encoding="utf-8")
    out = convert_portuguese_accents(raw)
    out = fix_spurious_accent_spaces(out)
    # Known corruption fixes from a first buggy pass
    out = out.replace("\\textbf{nõ}", "\\textbf{não}")
    out = out.replace("por si sás", "por si só")
    path.write_text(out, encoding="utf-8", newline="\n")
    print(f"Updated {path} (changed={raw != out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
