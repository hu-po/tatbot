# DrawingBotV3

- [DrawingBotV3](https://docs.drawingbotv3.com/en/latest/index.html)
- https://docs.drawingbotv3.com/en/latest/pensettings.html

Create an llm expert to answer questions by downloading docs, converting to markdown, and then uploading to "projects":

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


