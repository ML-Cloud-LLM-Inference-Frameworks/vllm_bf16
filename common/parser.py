from typing import Optional

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def parse_label(text: str) -> Optional[str]:
    if not text:
        return None

    x = text.strip().lower()

    for label in LABELS:
        if x == label.lower():
            return label

    candidates = []

    if "world" in x:
        candidates.append("World")
    if "sports" in x or "sport" in x:
        candidates.append("Sports")
    if "business" in x:
        candidates.append("Business")
    if (
        "sci/tech" in x
        or "science/technology" in x
        or "science and technology" in x
        or "technology" in x
        or "tech" in x
    ):
        candidates.append("Sci/Tech")

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0]

    return None