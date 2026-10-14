from __future__ import annotations

import json
import unittest

from locust_docs import LocustDocsRepository, parse_full_docs, parse_index, rank_documents
from server import McpServer


INDEX = """# Locust

## Getting started
- [Installation](https://docs.locust.io/en/stable/installation.html): Install Locust and troubleshoot common issues.
- [Distributed load generation](https://docs.locust.io/en/stable/running-distributed.html): Run a master and multiple workers.
- [API](https://docs.locust.io/en/stable/api.html): Python API reference.
"""

FULL = """# Locust

---
<a id="installation"></a>
# Installation

Install Locust with pip. On Windows, verify that the scripts directory is on PATH.

---
<a id="running-distributed"></a>
# Distributed load generation

Start one master process and connect multiple worker processes with --worker.

---
<a id="api"></a>
# API

## HttpUser class
Use HttpUser and its client attribute to make HTTP requests.

---
<a id="http-user-details"></a>
# HttpUser details

This unindexed Sphinx child content belongs to the API reference.
"""


class FakeRepository(LocustDocsRepository):
    def __init__(self) -> None:
        fixtures = {
            "https://docs.locust.io/en/stable/llms.txt": INDEX,
            "https://docs.locust.io/en/stable/llms-full.txt": FULL,
        }
        super().__init__(downloader=lambda url, timeout: fixtures[url], discover_html=False)


class ParsingTests(unittest.TestCase):
    def test_subsections_after_separator_are_preserved(self):
        text = FULL.replace('---\n<a id="running-distributed">',
                            '---\n## Troubleshooting\nMissing compiler solution.\n\n---\n<a id="running-distributed">')
        docs = {d.id: d for d in parse_full_docs(text, parse_index(INDEX))}
        self.assertIn('Missing compiler solution.', docs['installation'].text)
        self.assertNotIn('Missing compiler solution.', docs['running-distributed'].text)

    def test_discovery_adds_linked_guide_and_stays_in_prefix(self):
        base = 'https://docs.locust.io/en/stable'
        fixtures = {base + '/llms.txt': INDEX, base + '/llms-full.txt': FULL}
        fixtures[base + '/'] = '<div role="main"><a href="tasksets.html">Tasks</a><a href="https://example.com/evil.html">External</a><a href="../latest/other.html">Other version</a></div><footer></footer>'
        for name in ('installation', 'api', 'running-distributed'):
            fixtures[base + '/' + name + '.html'] = '<div role="main"><h1>Existing</h1></div><footer></footer>'
        fixtures[base + '/tasksets.html'] = '<div role="main"><h1>TaskSets</h1><h2>MarkovTaskSet</h2><pre>if ready:\n    run()</pre><a href="broken.html">Broken</a></div><footer>Ignore</footer>'
        visited = []
        def download(url, timeout):
            visited.append(url)
            return fixtures[url]
        repo = LocustDocsRepository(downloader=download)
        guide = repo.fetch('tasksets')
        self.assertIn('if ready:\n    run()', guide.text)
        self.assertNotIn('Ignore', guide.text)
        self.assertEqual(repo.search('MarkovTaskSet')[0].id, 'tasksets')
        self.assertTrue(all(url.startswith(base + '/') for url in visited))
        self.assertEqual(len(repo.coverage_warnings), 1)

    def test_index_and_full_docs_are_joined_by_stable_id(self) -> None:
        index = parse_index(INDEX)
        documents = parse_full_docs(FULL, index)
        by_id = {document.id: document for document in documents}

        self.assertEqual(by_id["installation"].title, "Installation")
        self.assertEqual(
            by_id["running-distributed"].url,
            "https://docs.locust.io/en/stable/running-distributed.html",
        )
        self.assertIn("HttpUser class", by_id["api"].text)
        self.assertIn("unindexed Sphinx child", by_id["api"].text)
        self.assertNotIn("http-user-details", by_id)

    def test_ranking_favors_a_title_and_full_query_match(self) -> None:
        documents = parse_full_docs(FULL, parse_index(INDEX))
        results = rank_documents("distributed workers", documents)
        self.assertEqual(results[0].id, "running-distributed")

    def test_ranking_matches_slug_punctuation(self) -> None:
        documents = parse_full_docs(FULL, parse_index(INDEX))
        results = rank_documents("running-distributed", documents)
        self.assertEqual(results[0].id, "running-distributed")


class RepositoryTests(unittest.TestCase):
    def test_repository_search_and_fetch(self) -> None:
        repository = FakeRepository()
        self.assertEqual(repository.search("HttpUser")[0].id, "api")
        self.assertIn("client attribute", repository.fetch("api").text)

    def test_unknown_document_fails(self) -> None:
        with self.assertRaises(KeyError):
            FakeRepository().fetch("missing")


class ProtocolTests(unittest.TestCase):
    def test_initialize_and_tool_listing(self) -> None:
        server = McpServer(FakeRepository())
        initialized = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            }
        )
        self.assertEqual(initialized["result"]["serverInfo"]["name"], "locust-docs")

        listed = server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(
            [tool["name"] for tool in listed["result"]["tools"]],
            ["search", "fetch", "list"],
        )

    def test_search_and_fetch_return_compatible_structured_content(self) -> None:
        server = McpServer(FakeRepository())
        searched = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": "worker processes"}},
            }
        )
        payload = searched["result"]["structuredContent"]
        self.assertEqual(payload["results"][0]["id"], "running-distributed")
        self.assertEqual(json.loads(searched["result"]["content"][0]["text"]), payload)

        fetched = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "fetch", "arguments": {"id": "running-distributed"}},
            }
        )
        self.assertIn("--worker", fetched["result"]["structuredContent"]["text"])

    def test_list_returns_every_loaded_document(self) -> None:
        server = McpServer(FakeRepository())
        listed = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "list", "arguments": {}},
            }
        )
        payload = listed["result"]["structuredContent"]
        self.assertEqual(payload["count"], 3)
        self.assertEqual([item["id"] for item in payload["documents"]], [
            "installation",
            "running-distributed",
            "api",
        ])

    def test_invalid_arguments_are_reported_as_json_rpc_errors(self) -> None:
        server = McpServer(FakeRepository())
        response = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "search", "arguments": {"query": ""}},
            }
        )
        self.assertEqual(response["error"]["code"], -32602)

        invalid_limit = server.dispatch(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": "api", "limit": 101},
                },
            }
        )
        self.assertEqual(invalid_limit["error"]["code"], -32602)


if __name__ == "__main__":
    unittest.main()
