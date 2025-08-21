---
summary: Rules for clean, minimal, agent-friendly docs
tags: [docs, style]
updated: 2025-08-21
audience: [dev, writer, agent]
---

# Documentation Style Guide

Short, minimal rules for writing docs that are easy to scan by humans and predictable for coding agents.

## ⚡ Core Principles

- **Minimal:** Short paragraphs (1–3 sentences). Prefer lists over prose.
- **Predictable:** Stable section names, consistent headings, tested commands.
- **Scannable:** H2 with emojis; use code blocks, short admonitions, and sparse grids.
- **Agent-friendly:** Include commands, paths, and config near the top; use anchors and cross-refs.

## 🧩 Page Template

Use this template as the default structure.

````markdown
---
summary: One-line purpose
tags: [topic, subtopic]
updated: YYYY-MM-DD
audience: [dev, agent]
---

# Page Title

One short intro sentence.

```{admonition} Quick Reference
:class: tip
- Command: `uv run …`
- Path: `src/tatbot/...`
- Config: `src/conf/...`
```

## ⚡ Quick Start
- Step 1 …
- Step 2 …

## 🛠️ Configuration
```yaml
# Prefer literalinclude in real docs
key: value
```

## Usage
```bash
uv run python -m tatbot.viz.teleop --enable-robot
```

## Troubleshooting
- Symptom → Fix
- Error → Cause → Verify command

## Reference
- :doc:`Related Topic <nodes>`
- :ref:`Internal Anchor`
````

## 🧭 Headings & Emojis

- **H1:** Page title only (one per page).
- **H2:** Major sections prefixed with a small emoji set: ⚡, 🛠️, 🖥️, 🦾, 📷, 🌐, 📚.
- **H3:** Subsections, no emoji. Avoid H4+; prefer lists or additional pages.

## 🎨 Visual Elements

- **Grids:** Use only for overviews/landing pages. Keep items concise.
- **Tabs:** Use for true alternatives (e.g., Method A vs B). Don’t tab trivial variations.
- **Admonitions:** Prefer built-ins: `{tip}`, `{note}`, `{warning}`, `{important}`. Keep content short.

Examples of fenced content inside examples must use four backticks:

````markdown
```{admonition} Prerequisites
:class: important
- Installed: `uv`, Python 3.11
```
````

## 🤖 Agent Optimization

- Put a “Quick Reference” block near the top (commands, paths, config).
- Use consistent H2 anchors: “Quick Start”, “Configuration”, “Usage”, “Troubleshooting”, “Reference”.
- Prefer structured lists over long paragraphs; avoid screenshots of text.
- Include real paths and commands; minimize interactive prompts and long outputs.

## 🔗 Linking & Reuse

- Prefer `:doc:` and `:ref:` for internal links; avoid raw file/line mentions.
- Use `literalinclude` to pull config/code from `src/` or `config/` to prevent drift.
- Use node substitutions defined in `conf.py` (e.g., `{{eek}}`) for consistent labels.

## ✅ Quality Checklist (keep it short)

- H1 once; H2 with approved emojis; H3 only as needed.
- Intro + Quick Reference present and accurate.
- Commands tested; code blocks have language tags.
- Links resolve; cross-refs use `:doc:`/`:ref:`.
- Page stays focused (< ~800 words). Split if needed.

## 🚫 Do / Don’t

- Do: Lists, short admonitions, stable anchors, `literalinclude`, minimal grids.
- Don’t: Emoji on H3+, decorative grids for simple lists, long terminal dumps, duplicate admonition styles.

## 🔧 Implementation Notes

- Theme: Furo with copy buttons and `sphinx_design`.
- MyST: `colon_fence`, `substitution`, anchors up to H3 as configured in `conf.py`.
- Keep substitutions, CSS, and theme tweaks centralized (see `docs/conf.py`, `_static/tatbot.css`).

