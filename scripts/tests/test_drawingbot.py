"""Geometry-preserving native acquisition contracts."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from drawingbot.artifacts import normalize_svg

HEADER = b'''<?xml version="1.0" encoding="ISO-8859-1"?>
<!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.0//EN'
'http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd'>'''
SVG = b'''<svg xmlns="http://www.w3.org/2000/svg" width="50.0mm" height="75.0mm">
<!-- export timestamp --><g id="black" transform="matrix(0.5,0,0,0.5,2,3)" style="stroke:black;fill:none">
<path d="M0 0 C10 0 10 20 20 20 M30 30 L40 40 Z"/></g></svg>'''


def test_normalization_preserves_geometry_and_absolute_page_scale():
    normalized, info = normalize_svg(HEADER + SVG)
    root = ET.fromstring(normalized)
    assert [float(n) for n in root.get("viewBox").split()] == pytest.approx([0, 0, 50 * 96 / 25.4, 75 * 96 / 25.4])
    original = ET.fromstring(SVG)
    assert ET.tostring(root[0]) == ET.tostring(original[0])
    assert info["raw_path_elements"] == 1
    assert "<!" not in normalized and "<?" not in normalized
    assert normalize_svg(normalized.encode())[0] == normalized


@pytest.mark.parametrize("prefix", [b'<!DOCTYPE svg [<!ENTITY x "boom">]>',
                                    b'<?xml-stylesheet href="outside.css"?>',
                                    b'<!DOCTYPE svg SYSTEM "file:///etc/passwd">'])
def test_normalizer_refuses_entities_and_unrecognized_xml(prefix):
    with pytest.raises(ValueError):
        normalize_svg(prefix + SVG)


def test_unsupported_geometry_is_retained_for_strict_native_decoder_refusal():
    svg = SVG.replace(b"</svg>", b'<image href="outside.png"/></svg>')
    assert "image" in normalize_svg(svg)[0]
