"""HumaneProxy CLI — developer-friendly command-line interface.

Install the package and run::

    humane-proxy init      # scaffold config + .env in your project
    humane-proxy start     # start the proxy server
    humane-proxy check "text"  # quick safety check from terminal
    humane-proxy version   # print version info
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import click

_BANNER = r"""
  _   _                                  ____
 | | | |_   _ _ __ ___   __ _ _ __   ___|  _ \ _ __ _____  ___   _
 | |_| | | | | '_ ` _ \ / _` | '_ \ / _ \ |_) | '__/ _ \ \/ / | | |
 |  _  | |_| | | | | | | (_| | | | |  __/  __/| | | (_) >  <| |_| |
 |_| |_|\__,_|_| |_| |_|\__,_|_| |_|\___|_|   |_|  \___/_/\_\\__, |
                                                                |___/
"""

_DEFAULT_YAML_CONTENT = """\
# HumaneProxy — project-level configuration.
# Values here override the package defaults.
# See: https://github.com/your-org/humane-proxy#configuration

server:
  host: "0.0.0.0"
  port: 8000
  reload: true          # auto-reload on code changes (dev mode)

safety:
  risk_threshold: 0.7
  spike_boost: 0.25

heuristics:
  self_harm_keywords:
    - "want to die"
    - "kill myself"
    - "end my life"
    - "suicide"
    - "suicidal"
    - "want to kill myself"
    - "can't go on"
    - "no reason to live"
    - "overdose on"
    - "slit my wrists"
    - "hang myself"
    - "cutting myself"
    - "hurt myself"
    - "self harm"
    - "self-harm"
    # Add your own keywords below:

  criminal_keywords:
    - "how to make a bomb"
    - "how to build a bomb"
    - "how to poison someone"
    - "how to make poison"
    - "how to kill someone"
    - "how to get away with murder"
    - "how to make meth"
    - "how to synthesize fentanyl"
    - "how to cook meth"
    - "child pornography"
    - "explosive device"
    - "how to make ricin"
    # Add your own keywords below:

  self_harm_keyword_score: 0.7
  criminal_keyword_score: 0.6
  intent_pattern_score: 0.7

  context_reducers:
    - "laughing"
    - "of laughter"
    - "of embarrassment"
    - "of boredom"
    - "in the game"
    - "in minecraft"
    - "for my character"
    - "in fiction"
    - "the villain"
    - "the character"
    - "in a novel"
    - "in a movie"
    - "in a book"
    - "in a story"
    - "my character"
    - "warning signs"
    - "prevent"
    - "prevention"
    - "how to help"
    - "help someone"
    - "help a friend"
    - "awareness"

trajectory:
  window_size: 5
  spike_delta: 0.35

escalation:
  rate_limit_max: 3
  rate_limit_window_hours: 1
  webhooks:
    slack_url: ""
    discord_url: ""
    pagerduty_routing_key: ""
"""

_DEFAULT_ENV_CONTENT = """\
# HumaneProxy environment variables.
# Rename this file to .env and fill in your values.

LLM_API_KEY=
LLM_API_URL=

# Optional overrides (uncomment to use):
# HUMANE_PROXY_PORT=8000
# HUMANE_PROXY_RISK_THRESHOLD=0.7
# HUMANE_PROXY_SLACK_URL=https://hooks.slack.com/services/...
# HUMANE_PROXY_DISCORD_URL=https://discord.com/api/webhooks/...
# HUMANE_PROXY_PAGERDUTY_KEY=your-routing-key
# HUMANE_PROXY_DB_PATH=/path/to/escalations.db
"""


@click.group()
def main() -> None:
    """🛡️  HumaneProxy — AI safety middleware that protects humans."""
    pass


@main.command()
def init() -> None:
    """Scaffold humane_proxy.yaml and .env.example in the current directory."""
    cwd = Path.cwd()
    created: list[str] = []

    yaml_path = cwd / "humane_proxy.yaml"
    if yaml_path.exists():
        click.echo(f"  ⚠  {yaml_path.name} already exists, skipping.")
    else:
        yaml_path.write_text(_DEFAULT_YAML_CONTENT, encoding="utf-8")
        created.append(yaml_path.name)

    env_path = cwd / ".env.example"
    if env_path.exists():
        click.echo(f"  ⚠  {env_path.name} already exists, skipping.")
    else:
        env_path.write_text(_DEFAULT_ENV_CONTENT, encoding="utf-8")
        created.append(env_path.name)

    if created:
        click.echo(f"\n  ✅ Created: {', '.join(created)}")
        click.echo("\n  Next steps:")
        click.echo("    1. Copy .env.example → .env and fill in your LLM_API_KEY / LLM_API_URL")
        click.echo("    2. Edit humane_proxy.yaml to customise thresholds & keywords")
        click.echo("    3. Run: humane-proxy start")
    else:
        click.echo("\n  ℹ  Nothing to create — files already exist.")


@main.command()
@click.option("--host", default=None, help="Bind host (default: from config)")
@click.option("--port", "-p", default=None, type=int, help="Bind port (default: from config)")
@click.option("--reload/--no-reload", default=None, help="Auto-reload on changes")
def start(host: str | None, port: int | None, reload: bool | None) -> None:
    """Start the HumaneProxy proxy server."""
    click.echo(_BANNER)

    from humane_proxy.config import get_config

    cfg = get_config()
    server_cfg = cfg.get("server", {})

    final_host = host or server_cfg.get("host", "0.0.0.0")
    final_port = port or server_cfg.get("port", 8000)
    final_reload = reload if reload is not None else server_cfg.get("reload", False)

    click.echo(f"  🛡️  Starting HumaneProxy on {final_host}:{final_port}")
    if final_reload:
        click.echo("  🔄 Auto-reload enabled")
    click.echo("")

    import uvicorn

    uvicorn.run(
        "humane_proxy.middleware.interceptor:app",
        host=final_host,
        port=final_port,
        reload=final_reload,
    )


@main.command()
@click.argument("text")
@click.option("--session", "-s", default="cli", help="Session ID for trajectory tracking")
def check(text: str, session: str) -> None:
    """Quick safety check on TEXT from the terminal."""
    from humane_proxy import HumaneProxy

    proxy = HumaneProxy()
    result = proxy.check(text, session_id=session)

    category = result.get("category", "safe")

    if category == "self_harm":
        icon = "🆘"
        label = "FLAGGED — self_harm"
    elif category == "criminal_intent" and not result["safe"]:
        icon = "⚠️"
        label = "FLAGGED — criminal_intent"
    elif result["safe"]:
        icon = "✅"
        label = "SAFE"
    else:
        icon = "⚠️"
        label = f"FLAGGED — {category}"

    click.echo(f"\n  {icon} {label}")
    click.echo(f"  Score   : {result['score']}")
    click.echo(f"  Category: {category}")
    if result["triggers"]:
        click.echo(f"  Triggers: {', '.join(result['triggers'])}")
    else:
        click.echo("  Triggers: (none)")
    click.echo("")


@main.command()
def version() -> None:
    """Print HumaneProxy version."""
    from humane_proxy import __version__
    click.echo(f"HumaneProxy v{__version__}")


@main.command()
@click.option("--category", "-c", default=None,
              help="Filter by category: self_harm | criminal_intent")
@click.option("--limit", "-n", default=20, type=int, help="Max records (default 20)")
@click.option("--session", "-s", default=None, help="Filter by session ID")
def escalations(category: str | None, limit: int, session: str | None) -> None:
    """List recent escalation events from the audit log."""
    import json
    import sqlite3
    from humane_proxy.escalation.local_db import _get_db_path

    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    try:
        clauses, params = [], []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if session:
            clauses.append("session_id = ?")
            params.append(session)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT id, session_id, category, risk_score, timestamp FROM escalations "
            f"{where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        click.echo("  ℹ  No escalations found.")
        return

    click.echo(f"\n  {'ID':<6} {'Session':<28} {'Category':<18} {'Score':<7} {'When'}")
    click.echo("  " + "-" * 75)
    for row in rows:
        id_, sid, cat, score, ts = row
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        emoji = "🆘" if cat == "self_harm" else "⚠️"
        click.echo(f"  {id_:<6} {sid:<28} {emoji} {cat:<16} {score:.2f}  {dt}")
    click.echo("")


@main.command()
@click.argument("session_id")
def session(session_id: str) -> None:
    """Show risk trajectory and escalation history for a session."""
    import json
    import sqlite3
    from humane_proxy.escalation.local_db import _get_db_path
    from humane_proxy.risk.trajectory import analyze

    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    try:
        rows = conn.execute(
            "SELECT category, risk_score, timestamp, triggers FROM escalations "
            "WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    click.echo(f"\n  📊 Session: {session_id}")
    click.echo(f"  Escalation count: {len(rows)}\n")

    if not rows:
        click.echo("  ℹ  No escalations recorded for this session.")
        return

    for cat, score, ts, triggers_json in rows:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        emoji = "🆘" if cat == "self_harm" else "⚠️"
        try:
            trigs = json.loads(triggers_json)
        except Exception:
            trigs = []
        click.echo(f"  {emoji} {dt}  score={score:.2f}  category={cat}")
        if trigs:
            click.echo(f"     triggers: {', '.join(trigs[:3])}")

    click.echo("")


@main.command("mcp-serve")
@click.option("--transport", "-t", default="stdio",
              type=click.Choice(["stdio", "http"]),
              help="Transport mode: stdio (default) or http")
@click.option("--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0)")
@click.option("--port", "-p", default=3000, type=int, help="HTTP bind port (default: 3000)")
def mcp_serve(transport: str, host: str, port: int) -> None:
    """Start the MCP server (requires [mcp] extra).

    Use --transport stdio (default) for local integration with agents.
    Use --transport http for remote access and registry listing.
    """
    try:
        if transport == "http":
            from humane_proxy.mcp_server import serve_http
            click.echo(f"  🤖 Starting HumaneProxy MCP server (HTTP) on {host}:{port}...")
            serve_http(host=host, port=port)
        else:
            from humane_proxy.mcp_server import serve
            click.echo("  🤖 Starting HumaneProxy MCP server (stdio)...")
            serve()
    except RuntimeError as exc:
        click.echo(f"\n  ❌ {exc}\n", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
