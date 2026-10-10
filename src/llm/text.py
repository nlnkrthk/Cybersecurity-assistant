"""
Text clean-up for model output.

Some models (e.g. gpt-oss) write non-breaking hyphens and narrow/thin spaces. They look the
same as plain characters but the Windows console (cp1252) can't print them, which shows as
'?' or crashes print(). They are replaced with their plain equivalents.
"""
_REPLACEMENTS = str.maketrans({
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "‒": "-",   # figure dash
    " ": " ",   # figure space
    " ": " ",   # thin space
    " ": " ",   # hair space
    " ": " ",   # narrow no-break space
    " ": " ",   # no-break space
})


def normalize_model_text(text: str) -> str:
    return text.translate(_REPLACEMENTS) if text else text
