"""
adjacency.py
============
Hardcoded neighbour map of Ukrainian oblasts (and Kyiv City), matching the
exact region labels in the dataset. Used to build spatial features without any
external geo / shapefile dependency, keeping the project fully reproducible.

Adjacency is real geographic border-sharing between oblasts. Kyiv City is
treated as adjacent to Kyivska oblast (it sits inside it). Permanent-siren
regions (Luhanska, Crimea) are intentionally absent.

Source for borders: standard administrative geography of Ukraine (oblast
border adjacency). Encoded manually and symmetric by construction below.
"""

from __future__ import annotations

# Directed pairs; we symmetrise programmatically afterwards.
_EDGES = [
    ("Kyiv City", "Kyivska oblast"),
    ("Kyivska oblast", "Zhytomyrska oblast"),
    ("Kyivska oblast", "Chernihivska oblast"),
    ("Kyivska oblast", "Cherkaska oblast"),
    ("Kyivska oblast", "Poltavska oblast"),
    ("Kyivska oblast", "Vinnytska oblast"),
    ("Chernihivska oblast", "Sumska oblast"),
    ("Sumska oblast", "Poltavska oblast"),
    ("Sumska oblast", "Kharkivska oblast"),
    ("Kharkivska oblast", "Poltavska oblast"),
    ("Kharkivska oblast", "Dnipropetrovska oblast"),
    ("Kharkivska oblast", "Donetska oblast"),
    ("Poltavska oblast", "Cherkaska oblast"),
    ("Poltavska oblast", "Dnipropetrovska oblast"),
    ("Poltavska oblast", "Kirovohradska oblast"),
    ("Cherkaska oblast", "Vinnytska oblast"),
    ("Cherkaska oblast", "Kirovohradska oblast"),
    ("Cherkaska oblast", "Kyivska oblast"),
    ("Vinnytska oblast", "Zhytomyrska oblast"),
    ("Vinnytska oblast", "Khmelnytska oblast"),
    ("Vinnytska oblast", "Odeska oblast"),
    ("Vinnytska oblast", "Kirovohradska oblast"),
    ("Zhytomyrska oblast", "Rivnenska oblast"),
    ("Zhytomyrska oblast", "Khmelnytska oblast"),
    ("Khmelnytska oblast", "Rivnenska oblast"),
    ("Khmelnytska oblast", "Ternopilska oblast"),
    ("Khmelnytska oblast", "Chernivetska oblast"),
    ("Rivnenska oblast", "Volynska oblast"),
    ("Rivnenska oblast", "Lvivska oblast"),
    ("Ternopilska oblast", "Lvivska oblast"),
    ("Ternopilska oblast", "Ivano-Frankivska oblast"),
    ("Ternopilska oblast", "Chernivetska oblast"),
    ("Lvivska oblast", "Volynska oblast"),
    ("Lvivska oblast", "Zakarpatska oblast"),
    ("Lvivska oblast", "Ivano-Frankivska oblast"),
    ("Ivano-Frankivska oblast", "Zakarpatska oblast"),
    ("Ivano-Frankivska oblast", "Chernivetska oblast"),
    ("Kirovohradska oblast", "Dnipropetrovska oblast"),
    ("Kirovohradska oblast", "Mykolaivska oblast"),
    ("Kirovohradska oblast", "Odeska oblast"),
    ("Dnipropetrovska oblast", "Zaporizka oblast"),
    ("Dnipropetrovska oblast", "Khersonska oblast"),
    ("Dnipropetrovska oblast", "Donetska oblast"),
    ("Donetska oblast", "Zaporizka oblast"),
    ("Zaporizka oblast", "Khersonska oblast"),
    ("Khersonska oblast", "Mykolaivska oblast"),
    ("Mykolaivska oblast", "Odeska oblast"),
]


def _build():
    adj: dict[str, set[str]] = {}
    for a, b in _EDGES:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {k: sorted(v) for k, v in adj.items()}


NEIGHBOURS = _build()


def neighbours_of(region: str) -> list[str]:
    """Return the list of oblasts bordering `region` (empty if unknown)."""
    return NEIGHBOURS.get(region, [])


if __name__ == "__main__":
    for r in sorted(NEIGHBOURS):
        print(f"{r:26s} -> {', '.join(NEIGHBOURS[r])}")
