"""Retrieval and ranking for Locust's AI-optimized documentation."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import logging
import os
import re
import threading
import time
from typing import Callable
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.request import Request, build_opener, HTTPRedirectHandler


DEFAULT_BASE_URL = "https://docs.locust.io/en/stable"
INDEX_FILE = "llms.txt"
FULL_FILE = "llms-full.txt"
USER_AGENT = "locust-docs-mcp/1.0 (+https://docs.locust.io/)"

_LINK_RE = re.compile(r"^- \[(?P<title>[^]]+)]\((?P<url>https://[^)]+)\)(?:: (?P<description>.*))?$")
_ANCHOR_RE = re.compile(r'^<a id="(?P<id>[^"]+)"></a>$', re.MULTILINE)
_TITLE_RE = re.compile(r"^# (?P<title>.+)$", re.MULTILINE)
# Treat punctuation used in slugs, CLI flags, and Python names as separators.
# This lets a query such as ``running-distributed`` match both the page slug and
# natural-language text such as ``running distributed``.
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    url: str
    description: str = ""


def _normalized(value: str) -> str:
    return " ".join(_WORD_RE.findall(value.lower()))


def _slug_from_url(url: str) -> str:
    filename = urlparse(url).path.rsplit("/", 1)[-1]
    return filename.removesuffix(".html") or "index"


def parse_index(text: str) -> list[Document]:
    """Parse Locust's llms.txt link index into document stubs."""
    documents: list[Document] = []
    for raw_line in text.splitlines():
        match = _LINK_RE.match(raw_line.strip())
        if not match:
            continue
        url = match.group("url")
        documents.append(
            Document(
                id=_slug_from_url(url),
                title=match.group("title").strip(),
                text="",
                url=url,
                description=(match.group("description") or "").strip(),
            )
        )
    return documents


def parse_full_docs(text: str, index_documents: list[Document]) -> list[Document]:
    """Split llms-full.txt into the source pages listed by llms.txt."""
    by_id = {document.id: document for document in index_documents}
    by_title = {_normalized(document.title): document for document in index_documents}
    parsed: list[Document] = []
    current: Document | None = None

    for raw_chunk in re.split(r"\n\s*---\s*\n", text):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        title_match = _TITLE_RE.search(chunk)
        title = title_match.group("title").strip() if title_match else ""
        anchor_match = _ANCHOR_RE.search(chunk)
        anchor_id = anchor_match.group("id") if anchor_match else ""
        stub = by_id.get(anchor_id) or by_title.get(_normalized(title))

        if stub:
            if current:
                parsed.append(current)
            current = Document(
                id=stub.id,
                title=title,
                text=chunk,
                url=stub.url,
                description=stub.description,
            )
        elif current:
            # Some Sphinx subpages appear as their own H1 blocks in llms-full.txt
            # but are not listed in llms.txt. Keep their content with the nearest
            # indexed parent so every exposed result retains a verified canonical URL.
            current = Document(
                id=current.id,
                title=current.title,
                text=f"{current.text}\n\n---\n\n{chunk}",
                url=current.url,
                description=current.description,
            )

    if current:
        parsed.append(current)

    # The corpus should have one entry per page. Keep the first on duplicate IDs,
    # because it carries the top-level page heading and canonical URL.
    unique: dict[str, Document] = {}
    for document in parsed:
        if document.id not in unique:
            unique[document.id] = document
            continue
        previous = unique[document.id]
        unique[document.id] = Document(
            id=previous.id,
            title=previous.title,
            text=f"{previous.text}\n\n---\n\n{document.text}",
            url=previous.url,
            description=previous.description,
        )
    return list(unique.values())


def rank_documents(query: str, documents: list[Document], limit: int = 8) -> list[Document]:
    """Rank pages with deterministic lexical scoring suitable for a small corpus."""
    phrase = _normalized(query)
    terms = list(dict.fromkeys(_WORD_RE.findall(query.lower())))
    if not terms:
        return []

    ranked: list[tuple[float, str, Document]] = []
    for document in documents:
        title = _normalized(document.title)
        description = _normalized(document.description)
        body = _normalized(document.text)
        url = _normalized(document.url)
        score = 0.0

        if phrase and phrase == title:
            score += 80
        elif phrase and phrase in title:
            score += 35
        if phrase and phrase in description:
            score += 12
        if phrase and phrase in body:
            score += 10

        matched_terms = 0
        for term in terms:
            title_hits = title.count(term)
            description_hits = description.count(term)
            body_hits = body.count(term)
            url_hits = url.count(term)
            if title_hits or description_hits or body_hits or url_hits:
                matched_terms += 1
            score += min(title_hits, 3) * 9
            score += min(description_hits, 3) * 4
            score += min(body_hits, 20) * 0.8
            score += min(url_hits, 2) * 5

        # Reward pages that cover the whole query, not just a frequent single term.
        score += (matched_terms / len(terms)) * 12
        if matched_terms == len(terms):
            score += 10
        if score > 0:
            ranked.append((score, title, document))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in ranked[:limit]]


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme != 'https' or parsed.hostname != 'docs.locust.io':
            raise ValueError('Redirect outside docs.locust.io refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HtmlPage(HTMLParser):
    """Extract Sphinx article text and links, preserving preformatted examples."""
    def __init__(self, html: str):
        super().__init__()
        self.active = False
        self.parts = []
        self.links = set()
        self.title_parts = []
        self.in_title = False
        self.feed(html)
        self.title = ''.join(self.title_parts).strip()
        self.text = ''.join(self.parts).strip()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('role') == 'main':
            self.active = True
        if tag == 'footer':
            self.active = False
        if not self.active:
            return
        if tag == 'a' and 'href' in attrs:
            self.links.add(attrs['href'])
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.parts.append('\n\n' + '#' * int(tag[1]) + ' ')
            self.in_title = tag == 'h1'
        elif tag == 'pre':
            self.parts.append('\n\n```\n')
        elif tag in ('p', 'div', 'ul', 'ol', 'dl', 'dt', 'tr'):
            self.parts.append('\n')
        elif tag == 'li':
            self.parts.append('\n- ')
        elif tag in ('td', 'th'):
            self.parts.append(' | ')

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_title = False
        if self.active and tag == 'pre':
            self.parts.append('\n```\n')

    def handle_data(self, data):
        if self.active:
            data = data.replace('\uf0c1', '').replace('¶', '')
            self.parts.append(data)
            if self.in_title:
                self.title_parts.append(data)


def _download_text(url: str, timeout: float) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "docs.locust.io":
        raise ValueError("Only HTTPS URLs on docs.locust.io are allowed")
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    with build_opener(SafeRedirect()).open(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset)


class LocustDocsRepository:
    """Thread-safe, time-bound in-memory cache of the public Locust docs."""

    def __init__(
        self,
        base_url: str | None = None,
        cache_ttl_seconds: int | None = None,
        timeout_seconds: float | None = None,
        downloader: Callable[[str, float], str] = _download_text,
        discover_html: bool = True,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCUST_DOCS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname != "docs.locust.io":
            raise ValueError("LOCUST_DOCS_BASE_URL must be an HTTPS URL on docs.locust.io")
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else int(os.getenv("LOCUST_DOCS_CACHE_TTL", "900"))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("LOCUST_DOCS_TIMEOUT", "20"))
        )
        self._downloader = downloader
        self.discover_html = discover_html
        self.coverage_warnings: list[str] = []
        self._documents: list[Document] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def _load(self) -> list[Document]:
        now = time.monotonic()
        if self._documents and now - self._loaded_at < self.cache_ttl_seconds:
            return self._documents
        with self._lock:
            now = time.monotonic()
            if self._documents and now - self._loaded_at < self.cache_ttl_seconds:
                return self._documents
            index_text = self._downloader(f"{self.base_url}/{INDEX_FILE}", self.timeout_seconds)
            full_text = self._downloader(f"{self.base_url}/{FULL_FILE}", self.timeout_seconds)
            index_documents = parse_index(index_text)
            documents = parse_full_docs(full_text, index_documents)
            if not documents:
                raise RuntimeError("Locust documentation corpus was empty or could not be parsed")
            if self.discover_html:
                documents = self._supplement_html(documents)
            self._documents = documents
            self._loaded_at = time.monotonic()
            return self._documents

    def _supplement_html(self, documents: list[Document]) -> list[Document]:
        """Discover linked HTML articles only within the configured version prefix."""
        prefix = self.base_url + '/'
        by_url = {d.url: d for d in documents}
        pending = {prefix, *by_url}
        seen = set()
        self.coverage_warnings = []

        def read(url):
            try:
                return url, HtmlPage(self._downloader(url, self.timeout_seconds)), None
            except Exception as exc:
                return url, None, str(exc)

        while pending and len(seen) < 100:
            batch = sorted(pending - seen)[:100 - len(seen)]
            if not batch:
                break
            pending -= set(batch)
            seen.update(batch)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for url, page, error in pool.map(read, batch):
                    if error:
                        self.coverage_warnings.append(f'{url}: {error}')
                        continue
                    for link in page.links:
                        target = urldefrag(urljoin(url, link))[0]
                        relative = target.removeprefix(prefix)
                        if (target.startswith(prefix) and target.endswith('.html')
                                and not relative.startswith(('_', 'search.', 'genindex.', 'py-modindex.'))
                                and target not in seen):
                            pending.add(target)
                    if url not in by_url and url != prefix and not url.endswith('/index.html'):
                        if page.title and page.text:
                            by_url[url] = Document(_slug_from_url(url), page.title, page.text, url)
                        else:
                            self.coverage_warnings.append(f'{url}: no article content extracted')
        if pending - seen:
            self.coverage_warnings.append('Discovery limit of 100 pages reached')
        for warning in self.coverage_warnings:
            logging.getLogger(__name__).warning('Documentation coverage: %s', warning)
        return list(by_url.values())

    def search(self, query: str, limit: int = 8) -> list[Document]:
        if not query.strip():
            return []
        return rank_documents(query, self._load(), limit=limit)

    def list_documents(self) -> list[Document]:
        """Return every page exposed by the currently loaded documentation corpus."""
        return list(self._load())

    def fetch(self, document_id: str) -> Document:
        requested = document_id.strip()
        if not requested:
            raise ValueError("Document ID is required")
        for document in self._load():
            if document.id == requested:
                return document
        raise KeyError(f"Unknown Locust documentation ID: {requested}")
