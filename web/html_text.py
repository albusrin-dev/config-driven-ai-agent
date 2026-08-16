"""HTML -> readable text, stdlib only.

Deliberately NOT a rendering or scripting engine: ``html.parser`` builds no
DOM, runs no JavaScript, fetches no subresources, and cannot execute
anything embedded in the page. Script/style/head content is dropped
outright, so injected payloads hiding there never even reach the model
(what does reach it is untrusted data either way, per Rule 12).

Raw HTML is token-heavy and injection-noisy; text is what the model needs.
If extraction quality ever demands boilerplate stripping (readability-style
main-content detection), that is the flagged upgrade — a library then, with
its supply-chain cost weighed openly.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Content of these elements is never text.
_DROP_CONTENT = {"script", "style", "head", "noscript", "template", "svg", "math"}
# These imply a line break in the extracted text.
_BLOCK = {
    "p", "div", "br", "li", "tr", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "ul",
    "ol", "nav", "main", "aside", "figure", "figcaption", "hr", "form",
}

_MANY_NEWLINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\r\f\v]+")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # entities decoded for us
        self.parts: list[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _DROP_CONTENT:
            self._drop_depth += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_CONTENT and self._drop_depth:
            self._drop_depth -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth == 0 and data.strip():
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    text = "".join(parser.parts)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()
