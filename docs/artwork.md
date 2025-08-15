# Artwork

Basic art workflow is:

1. image generation to create image (*.png)
2. vectorize into strokes (gcode)
3. convert gcode to IK poses (batched using GPU)

designs are stored in `nfs/designs`

## Image generation

- [Replicate Playground](https://replicate.com/playground)

## Vectorization

- [DrawingBotV3](https://docs.drawingbotv3.com/en/latest/index.html)
- tatbot uses the premium version
- [pen settings](https://docs.drawingbotv3.com/en/latest/pensettings.html)

custom `dbv3` config files:

- pens: `config/dbv3/pens/full.json`
- gcode: `config/dbv3/gcode/tatbot.json`
- areas: `config/dbv3/areas/fakeskin-landscape.json`

instructions to scrape docs for model context:

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