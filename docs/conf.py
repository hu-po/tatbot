import os
import sys

# Ensure package is importable if needed
sys.path.insert(0, os.path.abspath(".."))

project = "tatbot"
copyright = "2025, Hugo Ponte"
author = "Hugo Ponte"
version = "1.0"
release = "0.6.2"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "tasklist",
    "attrs_inline",
    "substitution",
]

# Generate HTML anchors for headings up to level 3 so in-page links work
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "CONSOLIDATION_PLAN.md"]

# Furo theme configuration
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["tatbot.css"]
html_favicon = "logos/favicon.ico"

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#000000",
        "color-brand-content": "#404040",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ffffff",
        "color-brand-content": "#b0b0b0",
        "color-sidebar-background": "#000000",
        "color-sidebar-brand-text": "#ffffff",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/hu-po/tatbot",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# MyST substitutions for reusable content
myst_substitutions = {
    "eek": "🦦 eek",
    "hog": "🦔 hog",
    "ook": "🦧 ook",
    "oop": "🦊 oop", 
    "ojo": "🦎 ojo",
    "rpi1": "🍓 rpi1",
    "rpi2": "🍇 rpi2",
}

# Support both .rst and .md sources
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Intersphinx configuration for cross-project references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}


