"""Read-only Ego2Robot progress MCP server for a private ChatGPT connection."""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from ego2robot.progress_reader import DOCUMENTS, ProgressReader


server = MCPServer("Ego2Robot Progress")


@server.tool()
def get_project_progress() -> dict[str, Any]:
    """Read the current project status, episode stage runs, scoped user reviews, and issue counts. Read only."""
    return ProgressReader().project_progress()


@server.tool()
def list_project_documents() -> dict[str, str]:
    """List the named project documents that can be read through this connector. Read only."""
    return dict(DOCUMENTS)


@server.tool()
def read_project_document(document: str, offset: int = 0, max_chars: int = 12000) -> dict[str, Any]:
    """Read a named design, plan, status, or review document in bounded pages. No arbitrary files. Read only."""
    return ProgressReader().read_document(document, offset, max_chars)


@server.tool()
def list_episodes() -> list[str]:
    """List available official episode directories under data/episodes. Read only."""
    return ProgressReader().list_episodes()


@server.tool()
def get_episode_progress(episode_id: str) -> dict[str, Any]:
    """Read stage run states, user reviews, and local evidence names for one episode. Read only."""
    return ProgressReader().episode_progress(episode_id)


@server.tool()
def get_open_issues() -> list[dict[str, str]]:
    """Read the explicit MOD checklist and priorities; checked means code change, not acceptance. Read only."""
    return ProgressReader().open_issues()


@server.tool()
def get_morphology_trial() -> dict[str, Any]:
    """Read summary metrics and review state of the UR5e, xArm7, and ViperX trials. Read only."""
    return ProgressReader().morphology_summary()


def main() -> None:
    """Run over stdio for Secure MCP Tunnel or another local MCP host."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
