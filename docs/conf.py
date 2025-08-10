import os
import sys

# Ensure package is importable if needed
sys.path.insert(0, os.path.abspath(".."))

project = "tatbot"
author = "Hugo Ponte"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# Generate HTML anchors for headings up to level 3 so in-page links work
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

# Support both .rst and .md sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


