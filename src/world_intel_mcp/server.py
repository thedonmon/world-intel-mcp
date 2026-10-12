#!/usr/bin/env python3
"""
World Intelligence MCP Server
==============================

Real-time global intelligence across 30 domains:
financial markets, economic indicators, earthquakes, wildfires,
conflict, military flights, infrastructure, and more.

Phase 1: Markets, Economic, Seismology, Wildfire (14 tools).
Phase 2: Conflict, Military, Infrastructure, Maritime, Climate (+10 = 24 tools).
Phase 3: News, Intelligence, Prediction, Displacement, Aviation, Cyber (+9 = 33 tools).
Phase 4: (reports removed — use live dashboard instead).
Phase 5: Analysis — focal points, signal summary, temporal anomalies, CII v2 (+3 = 39 tools).
Phase 6: Military & infrastructure intelligence (+6 = 45 tools).
Phase 7: Health, sanctions, elections, shipping, social, nuclear, alerts, trends (+10 = 55 tools).
Phase 8: Service status monitoring, RSS expansion (80+ feeds, 14 categories) (+1 = 56 tools).
Phase 9: Geospatial datasets — military bases, ports, pipelines, nuclear facilities (+4 = 60 tools).
Phase 10: NLP intelligence — entity extraction, event classification, news clustering, keyword spikes (+4 = 64 tools).
Phase 11: Strategic synthesis — strategic posture, world brief, fleet report, population exposure (+4 = 68 tools).
Phase 12: Extended geospatial (cables, datacenters, spaceports, minerals, exchanges), country stocks,
          aircraft batch, Hacker News, GitHub trending, arXiv papers, USA spending,
          NASA EONET, GDACS disaster alerts (+14 = 82 tools).
Phase 13: USNI fleet tracker, RSS expansion, report removal.
Phase 14: BTC technicals, central bank rates, trade routes, cloud regions, financial centers (+5 = 87 tools).
Phase 15: Business intelligence — forex (3), bonds/yields (2), earnings (2), SEC filings (3),
          company enrichment (1), macro composite (1) (+12 = 99 tools).
Phase 16: Vector intelligence — semantic search, similar events, timeline, vector stats,
          on-demand collection (+5 = 104 tools). Qdrant vector store auto-populates from all fetches.
          Collector daemon for 24/7 data accumulation. Enterprise-grade semantic retrieval.
Phase 17: Cross-domain analytics — cross-domain correlation, domain summary, trend detection
          (+3 = 109 tools). Historical analysis and early warning from accumulated vector data.
Phase 18: PDF/HTML intelligence reports (+1 = 110 tools). WeasyPrint-based multi-section
          report generation covering 18 intelligence domains in parallel.
Phase 19: Consumer energy signals (+3 = 113 tools). Retail fuel, residential natural gas,
          and electricity rates round out consumer energy monitoring.
Phase 20: Cited situation briefs and intel_daily_digest (+1 = 114 tools). Situation briefs
          now carry a numbered sources list and an honest cited flag; the new digest tool
          composes a cited markdown morning brief, degrading via data_gaps when the vector
          store is unavailable.
Phase 21: AOI geofences (+5 = 119 tools). intel_aoi_define/list/delete persist named
          point-radius areas in a dedicated table of the shared cache database.
          intel_aoi_brief composes a cited, radius-filtered view of earthquakes, military
          flights, wildfires, ACLED conflict events, sampled aviation, nearby static
          infrastructure, and news mentions for a user's own area, with data_gaps for
          domains that can't be scoped. intel_aoi_escalation reuses the existing hotspot
          scoring unmodified for a user AOI (#16).
Phase 22: intel_situation_brief (+1 = 120 tools). The cited situation brief (#15) is now
          reachable directly over MCP, not only through the dashboard: a bounded
          server-side gather of ~8-10 domains feeds the existing, unmodified
          fetch_situation_brief() (#18).
Phase 23: Geofence hardening + change detection (+2 = 122 tools). AOI bounding boxes
          split at the antimeridian instead of clamping (military fetches merge per-box
          with icao24 dedup; one-box failures surface as partial-coverage data_gaps);
          pipelines and undersea cables match on great-circle segment distance, not
          endpoints only. intel_aoi_update renames/resizes an AOI in place;
          intel_aoi_changes diffs each sweep against a stored per-AOI snapshot to
          report what entered and left the fence, with an explicit baseline first run
          and failed domains excluded from the diff rather than read as departures.
Phase 26: the 2,900-line registry/dispatch monolith split into domain modules
          under tools/ (12 modules, aggregated with an import-time
          TOOLS/HANDLERS parity guarantee) and shared infrastructure in
          runtime.py. Pure move: 128 tools, byte-identical definitions,
          identical dispatch semantics.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import click
import jsonschema

from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
    TextContent,
)

logging.basicConfig(
    level=os.environ.get("WORLD_INTEL_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("world-intel-mcp")

# Shared infrastructure lives in runtime.py (Phase 26 split); the local
# aliases remain as the module's public face — external readers (and the
# registry tests) reach the live cache and vector store as server.cache,
# server._vector_store, etc.
from . import runtime as _runtime  # noqa: E402
from . import tools as _tools_pkg  # noqa: E402

cache = _runtime.cache
breaker = _runtime.breaker
_aoi_store = _runtime.aoi_store
_vector_store = _runtime.vector_store
fetcher = _runtime.fetcher

# ---------------------------------------------------------------------------
# Tool registry — aggregated from the domain modules under tools/
# (Phase 26 split; per-module TOOLS/HANDLERS parity is enforced at
# import time by tools.aggregate).
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = list(_tools_pkg.ALL_TOOLS)
_TOOL_SCHEMAS = {tool.name: tool.input_schema for tool in TOOLS}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


async def _dispatch(name: str, arguments: dict[str, Any]) -> Any:
    """Route tool call to its domain-module handler."""
    handler = _tools_pkg.ALL_HANDLERS.get(name)
    if handler is not None:
        return await handler(arguments)
    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# MCP handlers
# ---------------------------------------------------------------------------


async def list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)


async def call_tool(
    ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    args = params.arguments or {}
    logger.info("Tool call: %s(%s)", params.name, json.dumps(args, default=str)[:200])
    try:
        if schema := _TOOL_SCHEMAS.get(params.name):
            jsonschema.validate(args, schema)
        result = await _dispatch(params.name, args)
        return CallToolResult(
            content=[TextContent(text=json.dumps(result, indent=2, default=str))]
        )
    except jsonschema.ValidationError as exc:
        return CallToolResult(
            is_error=True,
            content=[TextContent(text=f"Input validation error: {exc.message}")],
        )
    except Exception as exc:
        logger.exception("Tool failed: %s", params.name)
        return CallToolResult(is_error=True, content=[TextContent(text=str(exc))])


@asynccontextmanager
async def lifespan(app: Server):
    logger.info("World Intelligence MCP Server starting (%d tools)", len(TOOLS))
    if _vector_store:
        await _vector_store.start()
        logger.info("Vector store worker started")
    try:
        yield
    finally:
        if _vector_store:
            await _vector_store.stop()


server = Server(
    "world-intel-mcp",
    lifespan=lifespan,
    on_list_tools=list_tools,
    on_call_tool=call_tool,
    get_tool_input_schema=_TOOL_SCHEMAS.get,
)


def create_http_app(host: str = "127.0.0.1"):
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(request):
        return JSONResponse({"status": "ok"})

    return server.streamable_http_app(
        host=host,
        stateless_http=True,
        json_response=True,
        custom_starlette_routes=[Route("/healthz", health)],
    )


async def _run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    envvar="WORLD_INTEL_MCP_TRANSPORT",
    show_default=True,
)
@click.option(
    "--host", default="127.0.0.1", envvar="WORLD_INTEL_MCP_HOST", show_default=True
)
@click.option(
    "--port",
    default=8000,
    type=click.IntRange(1, 65535),
    envvar="WORLD_INTEL_MCP_PORT",
    show_default=True,
)
def run(transport: str, host: str, port: int) -> None:
    """Serve MCP over stdio or Streamable HTTP at /mcp."""
    if transport == "stdio":
        asyncio.run(_run())
    else:
        try:
            import uvicorn
        except ImportError as exc:
            raise click.ClickException(
                'Install HTTP support with pip install ".[http]"'
            ) from exc
        uvicorn.run(create_http_app(host), host=host, port=port)


if __name__ == "__main__":
    run()
