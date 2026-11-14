"""config/inkmap/inks.json: each ink's id, name and colour, for the public Inkmap. The catalogue's vendors, products and
stock stay in config/inks.yaml, which the public tree does not carry."""
import json
from pathlib import Path

from tatbot_art.inks import load_catalog, look

REPO = Path(__file__).resolve().parents[2]
INKS = json.loads((REPO / "config/inkmap/inks.json").read_text())["inks"]


def test_it_offers_every_ink_the_stock_designs_load():
    stock = json.loads((REPO / "web/inkmap/public/designs/manifest.json").read_text())["designs"]
    assert {ink for design in stock for ink in design["palette"]["inks"]} <= {ink["id"] for ink in INKS}


def test_it_offers_every_catalogue_ink_with_the_colour_a_separation_matches():
    catalog = load_catalog()
    assert {ink["id"] for ink in INKS} == set(catalog) - {"water"}, "water rinses; it is never a design's ink"
    for ink in INKS:
        seen = look(catalog, ink["id"])
        assert ink["name"] == seen.name and ink["rgb"] == [round(c * 255) for c in seen.srgb], ink["id"]
