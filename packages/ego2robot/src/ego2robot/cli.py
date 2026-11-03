"""``e2r`` command line interface.

Commands required by implementation-plan.md M0::

    e2r inspect <episode>
    e2r run <episode> --profile mvp --until <stage>
    e2r validate <episode>
    e2r viz <episode> --view <name>

plus the fixture tooling used to build the validation cases
(``e2r fixtures``) and the single-episode EgoDex fetch (``e2r fetch``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from . import __version__
from .core import artifacts as A
from .core.config import ConfigError, PipelineConfig, config_hash, resolve_config_from_cli
from .core.frames import Frame, transform_name
from .core.schema import EpisodeManifest, describe_pose_variable
from .pipeline.registry import default_registry
from .pipeline.runner import PipelineRunner

app = typer.Typer(
    name="e2r",
    help="Ego2Robot data pipeline: ego video -> canonical robot samples.",
    add_completion=False,
    no_args_is_help=True,
)
fixtures_app = typer.Typer(help="Build and inspect the validation cases.", no_args_is_help=True)
app.add_typer(fixtures_app, name="fixtures")

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return A.find_repo_root(Path.cwd())


def _resolve_episode_dir(episode: str | Path, cfg_episode_root: Path | None = None) -> Path:
    """Accept an episode directory or an episode id under the data root."""
    p = Path(episode)
    if p.is_dir():
        return p.resolve()
    if cfg_episode_root is not None:
        candidate = cfg_episode_root / str(episode)
        if candidate.is_dir():
            return candidate.resolve()
    raise typer.BadParameter(f"episode directory not found: {episode}")


def _load_config(
    *,
    profile: str,
    episode_id: str,
    repo_root: Path,
    source_profile: Optional[str] = None,
    robot_profile: Optional[str] = None,
    quality_profile: Optional[str] = None,
    sets: Optional[list[str]] = None,
    case: Optional[dict[str, Any]] = None,
) -> PipelineConfig:
    overrides: dict[str, Any] = {}
    for item in sets or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _coerce(value.strip())
    try:
        return resolve_config_from_cli(
            repo_root,
            profile=profile,
            source_profile=source_profile,
            robot_profile=robot_profile,
            quality_profile=quality_profile,
            set_values=overrides,
            case=case,
            episode_id=episode_id,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _coerce(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _runner(repo_root: Path) -> PipelineRunner:
    registry = default_registry()
    return PipelineRunner(registry.as_mapping())


# ---------------------------------------------------------------------------
# info commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"ego2robot {__version__}")
    console.print(f"python {sys.version.split()[0]}")


@app.command()
def frames() -> None:
    """Print the coordinate-frame registry and the transform naming rule."""
    from .core.frames import CONVENTIONS

    table = Table(title="coordinate frames", show_lines=False)
    table.add_column("frame", style="bold cyan")
    table.add_column("meaning")
    table.add_column("x")
    table.add_column("y")
    table.add_column("z")
    for f, conv in CONVENTIONS.items():
        table.add_row(f.value, conv.description, conv.x_axis, conv.y_axis, conv.z_axis)
    console.print(table)
    console.print(
        Panel(
            "T_A_B is the pose of frame B expressed in frame A.\n"
            "A point transforms as p_A = T_A_B @ p_B, so the *right* subscript is "
            "the source frame.\n"
            "Composition: T_A_C = T_A_B @ T_B_C.",
            title="transform naming (system-design.md 5.2)",
        )
    )
    console.print(
        f"camera->base variable: {transform_name(Frame.C, Frame.B)}   "
        f"base->TCP variable: {transform_name(Frame.B, Frame.TCP)}"
    )


@app.command()
def stages() -> None:
    """List the registered pipeline stages and the DAG order."""
    runner = _runner(_repo_root())
    registry = default_registry()
    dag = runner.dag()
    console.print(registry.describe())
    warnings = getattr(registry, "import_warnings", [])
    for w in warnings:
        err_console.print(f"[yellow]stage module unavailable:[/yellow] {w}")
    console.print()
    console.print(dag.describe())


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    episode: str = typer.Argument(..., help="Episode directory"),
    show_frames: int = typer.Option(0, "--show-frames", help="Print N rows of the canonical table"),
) -> None:
    """Show what an episode directory currently contains."""
    root = _resolve_episode_dir(episode)
    layout = A.EpisodeLayout(root)
    console.print(Panel(f"[bold]{root}[/bold]", title=f"episode {root.name}"))

    tree = Tree(root.name)
    for path in sorted(root.iterdir()):
        if path.is_dir():
            branch = tree.add(f"[cyan]{path.name}/[/cyan]")
            for child in sorted(path.iterdir())[:12]:
                size = child.stat().st_size if child.is_file() else 0
                branch.add(f"{child.name} [dim]({size} B)[/dim]")
            extra = len(list(path.iterdir())) - 12
            if extra > 0:
                branch.add(f"[dim]... {extra} more[/dim]")
        else:
            tree.add(f"{path.name} [dim]({path.stat().st_size} B)[/dim]")
    console.print(tree)

    if layout.episode_json.is_file():
        episode_model = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
        tb = Table(title="timebase + camera", show_header=False)
        tb.add_row("frames", str(episode_model.timebase.num_frames))
        tb.add_row("nominal fps", f"{episode_model.timebase.nominal_fps:.4f}")
        tb.add_row("timebase source", episode_model.timebase.timebase_source)
        tb.add_row("duration", f"{episode_model.timebase.duration_s:.3f} s")
        tb.add_row("camera", f"{episode_model.camera.width}x{episode_model.camera.height}")
        tb.add_row("fx, fy", f"{episode_model.camera.fx:.3f}, {episode_model.camera.fy:.3f}")
        tb.add_row("cx, cy", f"{episode_model.camera.cx:.3f}, {episode_model.camera.cy:.3f}")
        tb.add_row("calibration", episode_model.camera.calibration_source)
        tb.add_row("capture frame", Frame.C.value)
        tb.add_row("task", episode_model.task.instruction or "-")
        tb.add_row("license", episode_model.source.license)
        tb.add_row("publishable", str(episode_model.source.publishable))
        console.print(tb)

    manifests = Table(title="stage manifests")
    manifests.add_column("stage", style="bold")
    manifests.add_column("status")
    manifests.add_column("version")
    manifests.add_column("duration")
    manifests.add_column("outputs")
    found = False
    for stage in ("ingest", "hand", "retarget", "robot", "render", "projection", "quality"):
        m = A.read_stage_manifest(layout.manifest_path(stage))
        if m is None:
            continue
        found = True
        style = "green" if m.status == "succeeded" else "red"
        manifests.add_row(
            m.stage,
            f"[{style}]{m.status}[/{style}]",
            m.stage_version,
            f"{m.duration_s:.2f}s",
            str(len(m.outputs)),
        )
    if found:
        console.print(manifests)
    else:
        console.print("[yellow]no stage manifests yet[/yellow]")

    from .core.schema import CanonicalActionSequence

    if layout.actions_npz.is_file() and show_frames:
        action = CanonicalActionSequence.load_npz(layout.actions_npz)
        table = Table(title=f"canonical T_C_TCP (first {show_frames} rows)")
        for col in ("i", "t [s]", "x", "y", "z", "width [m]", "valid", "fallback"):
            table.add_column(col)
        pos = action["tcp_position_C_m"][:, 0]
        width = action["gripper_width_m"][:, 0]
        valid = action["valid"][:, 0]
        fb = action["fallback_code"][:, 0]
        ts = action["timestamp_s"]
        for i in range(min(show_frames, action.num_frames)):
            table.add_row(
                str(i),
                f"{ts[i]:.4f}",
                f"{pos[i, 0]:+.4f}",
                f"{pos[i, 1]:+.4f}",
                f"{pos[i, 2]:+.4f}",
                f"{width[i]:.4f}" if width[i] == width[i] else "nan",
                "Y" if valid[i] else "n",
                str(int(fb[i])),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    episode: str = typer.Argument(..., help="Episode directory"),
    profile: str = typer.Option("mvp", "--profile", "-p"),
    until: Optional[str] = typer.Option(None, "--until", help="Stop after this stage"),
    force_stage: list[str] = typer.Option([], "--force-stage", help="Rerun this stage and downstream"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the DAG and exit"),
    source_profile: Optional[str] = typer.Option(None, "--source-profile"),
    robot_profile: Optional[str] = typer.Option(None, "--robot-profile"),
    quality_profile: Optional[str] = typer.Option(None, "--quality-profile"),
    set_values: list[str] = typer.Option([], "--set", help="Config override key=value"),
    case_json: Optional[str] = typer.Option(None, "--case-json", help="Inline case block as JSON"),
) -> None:
    """Run the pipeline for one episode."""
    repo_root = _repo_root()
    episode_dir = _resolve_episode_dir(episode, repo_root / "data" / "episodes")
    case = json.loads(case_json) if case_json else None
    cfg = _load_config(
        profile=profile,
        episode_id=episode_dir.name,
        repo_root=repo_root,
        source_profile=source_profile,
        robot_profile=robot_profile,
        quality_profile=quality_profile,
        sets=set_values,
        case=case,
    )
    layout = A.EpisodeLayout(episode_dir, robot_id=cfg.robot_binding.robot_id)
    runner = _runner(repo_root)

    console.print(
        Panel(
            f"episode: {cfg.episode_id}\nprofile: {cfg.profile}\nconfig hash: {config_hash(cfg)[:16]}\n"
            f"stages: {' -> '.join(cfg.stages)}",
            title="e2r run",
        )
    )
    if dry_run:
        console.print(runner.dag(cfg.stages).describe())

    outcome = runner.run(
        cfg,
        layout,
        order=cfg.stages,
        until=until,
        force=force_stage,
        dry_run=dry_run,
    )

    table = Table(title="stage results")
    table.add_column("stage", style="bold")
    table.add_column("status")
    table.add_column("time")
    table.add_column("outputs")
    for rec in outcome.records:
        style = {
            "succeeded": "green",
            "cached": "cyan",
            "failed": "red",
            "skipped": "dim",
        }.get(rec.status, "yellow")
        table.add_row(
            rec.stage,
            f"[{style}]{rec.status}[/{style}]",
            f"{rec.duration_s:.2f}s",
            str(len(rec.outputs)),
        )
    console.print(table)

    A.atomic_write_json(layout.logs_dir / f"run-{outcome.run_id}.json", outcome.to_json())

    if not outcome.ok:
        err_console.print(f"[red]run failed[/red]: {outcome.error}")
        raise typer.Exit(code=1)
    console.print(f"[green]run ok[/green] (run_id {outcome.run_id})")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    episode: str = typer.Argument(..., help="Episode directory"),
    verify_checksums: bool = typer.Option(False, "--checksums", help="Re-hash every artifact"),
) -> None:
    """Re-open every artifact, validate its schema, and print the QC verdict."""
    root = _resolve_episode_dir(episode)
    layout = A.EpisodeLayout(root)
    ok = True

    from .core.schema import (
        CanonicalActionSequence,
        HandSequence,
        RobotTrajectory,
    )

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Any) -> None:
        nonlocal ok
        try:
            detail = fn()
            checks.append((name, True, str(detail or "")))
        except Exception as exc:  # noqa: BLE001 - validation reports, never raises
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))
            ok = False

    def _episode() -> str:
        m = EpisodeManifest.model_validate(A.read_json(layout.episode_json))
        return f"{m.timebase.num_frames} frames, {m.camera.width}x{m.camera.height}"

    def _hand() -> str:
        seq = HandSequence.load_npz(layout.hand_npz)
        seq.validate().raise_if_bad()
        return f"{seq.num_frames} frames, valid {float(seq['valid'].mean()):.3f}"

    def _action() -> str:
        seq = CanonicalActionSequence.load_npz(layout.actions_npz)
        seq.validate().raise_if_bad()
        return f"{seq.num_frames} frames, valid {float(seq['valid'].mean()):.3f}"

    def _trajectory() -> str:
        seq = RobotTrajectory.load_npz(layout.trajectory_npz)
        seq.validate().raise_if_bad()
        return f"IK success {float(seq['ik_success'].mean()):.3f}"

    def _manifests() -> str:
        missing = []
        for stage in ("ingest", "hand", "retarget", "robot"):
            if A.read_stage_manifest(layout.manifest_path(stage)) is None:
                missing.append(stage)
        if missing:
            raise FileNotFoundError(f"no manifest for {missing}")
        return "ingest, hand, retarget, robot"

    def _quality() -> str:
        payload = A.read_json(layout.quality_json)
        verdict = payload.get("gate_verdict", {})
        return f"valid={payload.get('valid')} gate={verdict.get('pass')}"

    checks_to_run = [
        ("episode.json", _episode),
        ("stage manifests", _manifests),
        ("hand/sequence.npz", _hand),
        ("canonical/actions.npz", _action),
        ("robot/trajectory.npz", _trajectory),
        ("quality/report.json", _quality),
    ]
    for name, fn in checks_to_run:
        if name == "episode.json" and not layout.episode_json.is_file():
            checks.append((name, False, "missing"))
            ok = False
            continue
        if name.startswith("hand/") and not layout.hand_npz.is_file():
            checks.append((name, False, "missing"))
            ok = False
            continue
        if name.startswith("canonical/") and not layout.actions_npz.is_file():
            checks.append((name, False, "missing"))
            ok = False
            continue
        if name.startswith("robot/") and not layout.trajectory_npz.is_file():
            checks.append((name, False, "missing"))
            ok = False
            continue
        if name.startswith("quality/") and not layout.quality_json.is_file():
            checks.append((name, False, "missing"))
            ok = False
            continue
        check(name, fn)

    if verify_checksums:
        n = 0
        for path in layout.iter_artifacts():
            try:
                A.file_sha256(path)
                n += 1
            except Exception as exc:  # noqa: BLE001
                checks.append((f"sha256 {path.name}", False, str(exc)))
                ok = False
        checks.append(("checksums", True, f"{n} files hashed"))
        _ = n

    table = Table(title=f"validate {root.name}")
    table.add_column("check", style="bold")
    table.add_column("result")
    table.add_column("detail")
    for name, passed, detail in checks:
        table.add_row(name, "[green]ok[/green]" if passed else "[red]FAIL[/red]", detail)
    console.print(table)

    from .quality.preflight import run_preflight

    preflight = run_preflight(layout, episode_id=root.name)
    console.print(
        Panel(
            f"passed: {preflight.passed}\nreasons: {', '.join(preflight.reasons) or 'none'}",
            title="preflight",
        )
    )
    if not preflight.passed:
        ok = False
    if not ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------


@app.command()
def viz(
    episode: str = typer.Argument(..., help="Episode directory"),
    view: str = typer.Option(..., "--view", help="hand-overlay | projection | 3d | report"),
    frame: int = typer.Option(0, "--frame", help="Frame index for still views"),
) -> None:
    """Open or regenerate a debug view for a processed episode."""
    root = _resolve_episode_dir(episode)
    layout = A.EpisodeLayout(root)
    previews = layout.previews_dir

    known = {
        "hand-overlay": previews / "hand-overlay.mp4",
        "hand-3d": previews / "hand-3d.png",
        "projection": previews / "projection-overlay.mp4",
        "projection-3d": previews / "projection-3d.png",
        "projection-golden": previews / "projection-golden-markers.png",
        "report": layout.quality_json,
    }
    if view == "list":
        table = Table(title="available views")
        table.add_column("view")
        table.add_column("path")
        for name, path in known.items():
            table.add_row(name, f"{path.name} [green]ok[/green]" if path.is_file() else "[dim]absent[/dim]")
        for p in sorted(previews.glob("*")):
            if p.name not in {q.name for q in known.values()}:
                table.add_row(p.name, "[cyan]present[/cyan]")
        console.print(table)
        return

    if view == "report":
        if not layout.quality_json.is_file():
            raise typer.BadParameter("no quality report yet; run the pipeline first")
        console.print_json(json.dumps(A.read_json(layout.quality_json)))
        return

    if view not in known:
        raise typer.BadParameter(
            f"unknown view {view!r}; known: {', '.join(sorted(known))}, list"
        )
    path = known[view]
    if not path.is_file() and view.startswith("projection") and frame:
        candidate = previews / f"projection-frame-{frame:06d}.png"
        if candidate.is_file():
            path = candidate
    if not path.is_file():
        raise typer.BadParameter(
            f"{path} does not exist; run the pipeline (and the stage that produces it) first"
        )
    console.print(str(path))
    try:
        import os

        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception:
        console.print("[dim]open the path above in your viewer[/dim]")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@fixtures_app.command("list")
def fixtures_list(
    case_root: Optional[str] = typer.Option(None, "--root", help="Fixture output directory"),
) -> None:
    """List the synthetic validation cases defined in configs/fixtures/cases.yaml."""
    from .tools.cases import CASES_FILE, load_cases

    repo = _repo_root()
    root = Path(case_root) if case_root else repo / "data" / "fixtures"
    cases = load_cases(repo / CASES_FILE)
    table = Table(title=f"validation cases ({len(cases)})")
    for col in ("id", "task", "frames", "fps", "what it checks", "built"):
        table.add_column(col)
    for c in cases:
        built = (root / c.episode_id / "source" / "video.mp4").is_file()
        table.add_row(
            c.case_id,
            c.instruction,
            str(c.num_frames),
            f"{c.fps:g}",
            c.checks,
            "[green]yes[/green]" if built else "[dim]no[/dim]",
        )
    console.print(table)


@fixtures_app.command("build")
def fixtures_build(
    case_root: Optional[str] = typer.Option(None, "--root", help="Fixture output directory"),
    only: Optional[str] = typer.Option(None, "--only", help="Build a single case id"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if present"),
) -> None:
    """Generate the synthetic ego clips, HDF5 annotations and per-case config."""
    from .tools.cases import CASES_FILE, load_cases
    from .tools.synthetic import build_cases

    repo = _repo_root()
    root = Path(case_root) if case_root else repo / "data" / "fixtures"
    cases = load_cases(repo / CASES_FILE)
    if only:
        cases = [c for c in cases if c.case_id == only]
        if not cases:
            raise typer.BadParameter(f"no case with id {only!r}")
    built = build_cases(cases, root=root, force=force, console=console)
    console.print(f"[green]built {len(built)} case(s)[/green] under {root}")


@fixtures_app.command("verify")
def fixtures_verify(
    case_root: Optional[str] = typer.Option(None, "--root", help="Fixture output directory"),
    only: Optional[str] = typer.Option(None, "--only", help="Verify a single case id"),
    force_stage: list[str] = typer.Option([], "--force-stage"),
) -> None:
    """Run the pipeline over the selected cases and summarise the gates."""
    from .tools.run_cases import run_cases

    repo = _repo_root()
    root = Path(case_root) if case_root else repo / "data" / "fixtures"
    run_cases(
        repo_root=repo,
        fixture_root=root,
        only=only,
        force=force_stage,
        console=console,
    )


# ---------------------------------------------------------------------------
# fetch (EgoDex single episode)
# ---------------------------------------------------------------------------


@app.command()
def fetch(
    archive: str = typer.Option("test", "--archive", help="EgoDex archive name"),
    task: Optional[str] = typer.Option(None, "--task", help="Restrict to one task folder"),
    index: Optional[int] = typer.Option(None, "--index", help="Episode index inside the task"),
    dest: str = typer.Option(..., "--dest", help="Episode directory to create"),
    max_mb: float = typer.Option(80.0, "--max-mb", help="Preferred maximum episode size in MiB"),
    list_only: bool = typer.Option(False, "--list", help="List candidate tasks and exit"),
) -> None:
    """Download a single short episode from the public EgoDex archives."""
    from .ingest.egodex_download import (
        download_episode,
        open_archive,
        select_episode,
        summarise_sizes,
        EgoDexZipIndex,
    )

    console.print(f"opening archive [cyan]{archive}[/cyan] ...")
    arc = open_archive(archive)
    console.print(f"archive size {arc.size / (1024**3):.2f} GiB, range requests: {arc.supports_range}")
    if not arc.supports_range:
        raise typer.BadParameter(
            "this server does not honour HTTP range requests; a selective fetch is impossible"
        )
    idx = EgoDexZipIndex(arc)
    refs = idx.episodes(tasks=[task] if task else None)
    console.print(f"{len(refs)} paired episode(s) found")
    if list_only:
        for line in summarise_sizes(refs, limit=25):
            console.print(line)
        return

    ref = select_episode(refs, task=task, index=index, max_total_bytes=int(max_mb * 1024 * 1024))
    console.print(f"selected [bold]{ref.name}[/bold]: {ref.describe()}")
    dest_dir = Path(dest)
    hdf5_path, mp4_path = download_episode(ref, idx, dest_dir / "source")
    console.print(f"[green]downloaded[/green] {hdf5_path.name}, {mp4_path.name} into {dest_dir / 'source'}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
