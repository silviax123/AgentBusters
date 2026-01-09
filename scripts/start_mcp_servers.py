#!/usr/bin/env python3
"""
Start MCP Servers for AgentBusters

Starts all three MCP servers (SEC EDGAR, Yahoo Finance, Sandbox)
using FastMCP's built-in server capabilities.
"""

import argparse
import asyncio
import sys
from datetime import datetime

sys.path.insert(0, "src")

from mcp_servers.sec_edgar import create_edgar_server
from mcp_servers.yahoo_finance import create_yahoo_finance_server
from mcp_servers.sandbox import create_sandbox_server


def parse_args():
    parser = argparse.ArgumentParser(description="Start MCP servers for AgentBusters")
    parser.add_argument(
        "--server",
        choices=["edgar", "yfinance", "sandbox", "all"],
        default="all",
        help="Which server to start (default: all)",
    )
    parser.add_argument(
        "--simulation-date",
        type=str,
        help="Simulation date for temporal locking (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse simulation date if provided
    simulation_date = None
    if args.simulation_date:
        simulation_date = datetime.fromisoformat(args.simulation_date)
        print(f"Temporal locking enabled: simulation date = {simulation_date}")

    transport_kwargs = {}
    if args.transport != "stdio":
        transport_kwargs = {"host": args.host, "port": args.port}

    if args.server == "edgar":
        print("Starting SEC EDGAR MCP server...")
        server = create_edgar_server(simulation_date=simulation_date)
        server.run(transport=args.transport, **transport_kwargs)

    elif args.server == "yfinance":
        print("Starting Yahoo Finance MCP server...")
        server = create_yahoo_finance_server(simulation_date=simulation_date)
        server.run(transport=args.transport, **transport_kwargs)

    elif args.server == "sandbox":
        print("Starting Python Sandbox MCP server...")
        server = create_sandbox_server()
        server.run(transport=args.transport, **transport_kwargs)

    elif args.server == "all":
        print("To run all servers, start each in a separate terminal:")
        print()
        print("Terminal 1: python scripts/start_mcp_servers.py --server edgar")
        print("Terminal 2: python scripts/start_mcp_servers.py --server yfinance")
        print("Terminal 3: python scripts/start_mcp_servers.py --server sandbox")
        print()
        print("Or use the MCP Inspector to test individual servers:")
        print("  npx @anthropic-ai/mcp-inspector python scripts/start_mcp_servers.py --server edgar")


if __name__ == "__main__":
    main()
