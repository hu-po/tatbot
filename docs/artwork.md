---
summary: Artwork pipeline from image to robot strokes
tags: [artwork, pipeline]
updated: 2025-08-21
audience: [dev, operator, agent]
---

# 🎨 Artwork

Basic art workflow is:

1. image generation to create image (*.png)
2. vectorize into strokes (gcode)
3. convert gcode to IK poses (batched using GPU)

designs are stored in `/nfs/tatbot/designs/`

```{admonition} Quick Reference
:class: tip
- Designs: `/nfs/tatbot/designs/`
- DBv3 configs: `config/dbv3/{pens,gcode,areas}/...`
```

## Image Generation

- [Replicate Playground](https://replicate.com/playground)

## Vectorization

- [DrawingBotV3](https://docs.drawingbotv3.com/en/latest/index.html)
- tatbot uses the premium version
- [pen settings](https://docs.drawingbotv3.com/en/latest/pensettings.html)

custom `dbv3` config files:

- pens: `config/dbv3/pens/full.json`
- gcode: `config/dbv3/gcode/tatbot.json`
- areas: `config/dbv3/areas/fakeskin-landscape.json`

### Docs to Markdown (optional)
Instructions to scrape docs for model context:

```bash
cd /tmp
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent https://docs.drawingbotv3.com/en/latest/
uv venv
source .venv/bin/activate
uv pip install html2text
find docs.drawingbotv3.com -name '*.html' -exec sh -c 'uv run html2text "{}" > "${0%.html}.md"' {} \;
# Create project folder with only markdown files
mkdir docs.drawingbotv3.com.project
find docs.drawingbotv3.com -name '*.md' -exec cp {} docs.drawingbotv3.com.project/ \;
```
