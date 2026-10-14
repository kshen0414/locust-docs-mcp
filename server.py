#!/usr/bin/env python3
"""Dependency-free STDIO MCP server for the Locust documentation."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from locust_docs import LocustDocsRepository


SERVER_NAME = "locust-docs"
SERVER_VERSION = "1.2.0"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"

INSTRUCTIONS = (
    "Use search to find relevant pages in the current stable Locust documentation, "
    "or list to inspect the complete loaded corpus, then fetch a result by ID before answering. "
    "Prefer the returned canonical URL "
    "for citations. This server is read-only and retrieves docs.locust.io content only."
)

SEARCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "url": {"type": "string", "format": "uri"},
                },
                "required": ["id", "title", "url"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

FETCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "text": {"type": "string"},
        "url": {"type": "string", "format": "uri"},
        "metadata": {"type": "object"},
    },
    "required": ["id", "title", "text", "url", "metadata"],
    "additionalProperties": False,
}

LIST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "warnings": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer", "minimum": 0},
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "url": {"type": "string", "format": "uri"},
                },
                "required": ["id", "title", "description", "url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["count", "documents"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "search",
        "title": "Search Locust documentation",
        "description": (
            "Search the current stable Locust documentation for concepts, APIs, configuration, "
            "examples, and troubleshooting guidance. Call fetch on a returned ID for full content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 8},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "outputSchema": SEARCH_OUTPUT_SCHEMA,
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "fetch",
        "title": "Fetch a Locust documentation page",
        "description": (
            "Retrieve the full Markdown content and canonical citation URL for a Locust docs result. "
            "Use an ID returned by search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string", "minLength": 1}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "outputSchema": FETCH_OUTPUT_SCHEMA,
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
    {
        "name": "list",
        "title": "List Locust documentation pages",
        "description": (
            "List every documentation page loaded from the current Locust documentation corpus. "
            "Use this to inspect coverage or discover stable IDs before calling fetch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "outputSchema": LIST_OUTPUT_SCHEMA,
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True,
        },
    },
]


class McpServer:
    def __init__(self, repository: LocustDocsRepository | None = None) -> None:
        self.repository = repository or LocustDocsRepository()

    @staticmethod
    def _tool_result(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "structuredContent": value,
            "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
            "isError": False,
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("search requires a non-empty string argument named 'query'")
            limit = arguments.get("limit", 8)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                raise ValueError("search 'limit' must be an integer between 1 and 100")
            results = [
                {"id": document.id, "title": document.title, "url": document.url}
                for document in self.repository.search(query, limit=limit)
            ]
            return self._tool_result({"results": results})
        if name == "fetch":
            document_id = arguments.get("id")
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("fetch requires a non-empty string argument named 'id'")
            document = self.repository.fetch(document_id)
            return self._tool_result(
                {
                    "id": document.id,
                    "title": document.title,
                    "text": document.text,
                    "url": document.url,
                    "metadata": {
                        "source": "docs.locust.io",
                        "version": "stable",
                        "content_format": "markdown",
                    },
                }
            )
        if name == "list":
            documents = [
                {
                    "id": document.id,
                    "title": document.title,
                    "description": document.description,
                    "url": document.url,
                }
                for document in self.repository.list_documents()
            ]
            return self._tool_result({"count": len(documents), "documents": documents,
                                      "warnings": self.repository.coverage_warnings})
        raise KeyError(f"Unknown tool: {name}")

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")

        # Notifications intentionally do not receive JSON-RPC responses.
        if request_id is None:
            return None

        try:
            if method == "initialize":
                requested_version = request.get("params", {}).get("protocolVersion")
                result = {
                    "protocolVersion": requested_version or DEFAULT_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params") or {}
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not isinstance(name, str) or not isinstance(arguments, dict):
                    raise ValueError("tools/call requires a tool name and object arguments")
                result = self.call_tool(name, arguments)
            else:
                return _error_response(request_id, -32601, f"Method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (ValueError, KeyError) as exc:
            return _error_response(request_id, -32602, str(exc))
        except Exception as exc:  # Keep transport alive and keep details off stdout.
            print(f"{SERVER_NAME}: {exc}", file=sys.stderr, flush=True)
            if "--debug" in sys.argv:
                traceback.print_exc(file=sys.stderr)
            return _error_response(request_id, -32603, "Internal error while reading Locust documentation")


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def run_stdio(server: McpServer | None = None) -> None:
    # MCP STDIO is UTF-8 JSON Lines. Force UTF-8 on Windows, where redirected
    # standard streams may otherwise inherit a legacy console code page.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    implementation = server or McpServer()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            response = implementation.dispatch(request)
        except (json.JSONDecodeError, ValueError) as exc:
            response = _error_response(None, -32700, f"Parse error: {exc}")
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio()
