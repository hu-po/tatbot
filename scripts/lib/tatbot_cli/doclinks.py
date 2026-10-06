"""Check local Markdown targets without importing a documentation toolchain."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


def _prose(text):
    return re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", text)


def anchors(text):
    found = set(re.findall(r"(?m)^\(([^)]+)\)=\s*$", text))
    found.update(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', text))
    counts = Counter()
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)(?:\s+#+)?$", _prose(text)):
        explicit = re.search(r"\{#([^}]+)\}", heading)
        if explicit:
            found.add(explicit[1])
        slug = re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")
        found.add(slug + (f"-{counts[slug]}" if counts[slug] else ""))
        counts[slug] += 1
    return found


def target_problem(repo: Path, source: Path, target: str):
    target = target.strip().split(' "', 1)[0].strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    path = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source.resolve()
    if not path.is_relative_to(repo.resolve()):
        return f"{source.relative_to(repo)}: local link leaves repository: {target}"
    if not path.exists():
        return f"{source.relative_to(repo)}: missing local target {target}"
    if parsed.fragment and path.suffix == ".md" and unquote(parsed.fragment) not in anchors(path.read_text()):
        return f"{source.relative_to(repo)}: missing local anchor {target}"
    return None


def check(repo: Path, paths):
    problems = []
    for source in sorted(set(paths)):
        if not source.is_file():
            problems.append(f"missing current documentation: {source.relative_to(repo)}")
            continue
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", _prose(source.read_text())):
            problem = target_problem(repo, source, target)
            if problem:
                problems.append(problem)
    return problems
