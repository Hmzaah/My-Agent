"""Command-line interface: jolt ingest | ask | serve | eval | graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from jolt.config import get_settings

if TYPE_CHECKING:
    from jolt.service import JoltService

app = typer.Typer(add_completion=False, help="Jolt: agentic RAG over your documents.")


def _setup_logging(verbose: bool = False) -> None:
    level = (
        logging.DEBUG
        if verbose
        else getattr(logging, get_settings().log_level.upper(), logging.INFO)
    )
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def _service() -> JoltService:
    from jolt.service import JoltService

    return JoltService.from_settings(get_settings())


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders (.pdf, .txt, .md)"),
    rebuild: bool = typer.Option(False, help="Drop the existing index first"),
) -> None:
    """Add documents to the local index."""
    _setup_logging()
    result = _service().ingest(paths, rebuild=rebuild)
    typer.echo(
        f"Indexed {result.files} files: {result.chunks_added} new chunks "
        f"({result.total_chunks} total)"
    )


@app.command()
def ask(
    question: str,
    web: bool = typer.Option(True, "--web/--no-web", help="Allow web search fallback"),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON response"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Ask a question and print the answer with its sources."""
    _setup_logging(verbose)
    result = _service().ask(question, use_web=web)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"\n{result.answer}\n")
    for c in result.citations:
        where = c.url or (f"{c.source}, p.{c.page}" if c.page else c.source)
        typer.echo(f"  [{c.index}] {where}")
    steps = " -> ".join(s.node for s in result.trace)
    typer.echo(f"\n{steps}  ({result.latency_ms:.0f} ms)")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the HTTP API."""
    import uvicorn

    _setup_logging()
    uvicorn.run("jolt.api:app", host=host, port=port, reload=reload)


@app.command("eval")
def evaluate(
    dataset: Path = typer.Argument(Path("eval/questions.jsonl"), exists=True),
    out: Path = typer.Option(Path("eval/results"), help="Folder for the report"),
    web: bool = typer.Option(False, "--web/--no-web", help="Allow web search during eval"),
) -> None:
    """Run the evaluation set and write a Markdown + JSON report."""
    from jolt.evaluation import run_evaluation

    _setup_logging()
    report = run_evaluation(_service(), dataset, out, use_web=web)
    typer.echo(report.to_markdown())


@app.command()
def graph() -> None:
    """Print the agent graph as a Mermaid diagram."""
    typer.echo(_service().graph_mermaid())


if __name__ == "__main__":
    app()
