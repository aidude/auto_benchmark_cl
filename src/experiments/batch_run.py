"""
Batch runner — expand experiments.yaml combos and execute in parallel.

Each experiment is a Cartesian product: tasks × models × systems.
Completed combos are auto-skipped (implicit resume).

Usage:
  python -m src.experiments.batch_run --list
  python -m src.experiments.batch_run --experiment horse_race_v1
  python -m src.experiments.batch_run --experiment horse_race_v1 --dry-run
  python -m src.experiments.batch_run --all

Cron (nightly at 2 am):
  0 2 * * * cd /path/to/auto_benchmark_cl && \\
            conda run -n twopac bash scripts/run_batch.sh nightly >> logs/cron.log 2>&1
"""
import argparse
import json
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.utils.logger import get_logger, RUN_ID
from src.experiments.run import (
    run, resolve_model, _check_key, SYSTEMS, TASKS, _PROJECT_ROOT,
)

_log = get_logger(__name__)

_EXPERIMENTS_YAML = _PROJECT_ROOT / "configs" / "experiments.yaml"


# ── Data structures ───────────────────────────────────────────

@dataclass
class Combo:
    task:    str
    model:   str
    system:  str


@dataclass
class ComboResult:
    combo:   Combo
    status:  str          # "ok" | "failed" | "skipped"
    elapsed: float = 0.0
    overall_mean: float | None = None
    output:  str = ""
    error:   str = ""


# ── Helpers ───────────────────────────────────────────────────

def _load_experiments() -> dict:
    return yaml.safe_load(_EXPERIMENTS_YAML.read_text()).get("experiments", {})


def _expand(exp_cfg: dict) -> list[Combo]:
    combos = []
    for task in exp_cfg["tasks"]:
        for model in exp_cfg["models"]:
            for system in exp_cfg.get("systems", ["stateless"]):
                if task not in TASKS:
                    _log.warning("Unknown task '%s' — skipping", task)
                    continue
                if system not in SYSTEMS:
                    _log.warning("Unknown system '%s' — skipping", system)
                    continue
                combos.append(Combo(task=task, model=model, system=system))
    return combos


def _outpath(outdir: Path, combo: Combo) -> Path:
    slug = combo.model.replace("/", "-")
    return outdir / f"{combo.task}_{slug}_{combo.system}.json"


# ── Single-combo runner ───────────────────────────────────────

def _run_combo(combo: Combo, outdir: Path) -> ComboResult:
    out = _outpath(outdir, combo)

    if out.exists():
        # Only skip if the file is a complete (non-partial) result.
        try:
            if json.loads(out.read_text()).get("complete", True):
                _log.info("SKIP  %s / %s / %s (result exists)", combo.task, combo.model, combo.system)
                return ComboResult(combo=combo, status="skipped", output=str(out))
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt / partial file → re-run

    t0 = time.time()
    try:
        info = resolve_model(combo.model)
        _check_key(info)
        # run() writes to outpath after every task (partial) and at completion (final)
        result = run(combo.system, info, combo.task, outpath=out)
        elapsed = round(time.time() - t0, 1)
        _log.info(
            "OK    %s / %s / %s  overall=%.4f  %.1fs",
            combo.task, combo.model, combo.system, result["overall_mean"], elapsed,
        )
        return ComboResult(
            combo=combo, status="ok", elapsed=elapsed,
            overall_mean=result["overall_mean"], output=str(out),
        )
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        _log.error("FAIL  %s / %s / %s — %s", combo.task, combo.model, combo.system, exc)
        return ComboResult(combo=combo, status="failed", elapsed=elapsed, error=str(exc))


# ── Experiment runner ─────────────────────────────────────────

def run_experiment(name: str, exp_cfg: dict, dry_run: bool = False) -> dict:
    combos     = _expand(exp_cfg)
    parallel   = exp_cfg.get("parallel", True)
    max_workers = min(exp_cfg.get("max_workers", 3), len(combos))
    date       = datetime.now().strftime("%Y-%m-%d")
    outdir     = _PROJECT_ROOT / "results" / date / name
    outdir.mkdir(parents=True, exist_ok=True)

    desc = exp_cfg.get("description", "")
    _log.info(
        "=== Experiment '%s' (%s) | %d combos | parallel=%s workers=%d ===",
        name, desc, len(combos), parallel, max_workers,
    )

    if dry_run:
        for c in combos:
            exists = "✓ exists" if _outpath(outdir, c).exists() else "  pending"
            print(f"  [{exists}]  {c.task} / {c.model} / {c.system}")
        return {"name": name, "dry_run": True, "combos": len(combos)}

    t0 = time.time()
    results: list[ComboResult] = []

    combo_bar = tqdm(total=len(combos), desc=f"[{name}]", unit="combo", dynamic_ncols=True)

    def _done(r: ComboResult) -> None:
        results.append(r)
        ok   = sum(1 for x in results if x.status == "ok")
        fail = sum(1 for x in results if x.status == "failed")
        skip = sum(1 for x in results if x.status == "skipped")
        combo_bar.set_postfix(ok=ok, fail=fail, skip=skip)
        combo_bar.update(1)

    if parallel and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_combo, c, outdir): c for c in combos}
            for fut in as_completed(futures):
                _done(fut.result())
    else:
        for combo in combos:
            _done(_run_combo(combo, outdir))

    combo_bar.close()

    elapsed  = round(time.time() - t0, 1)
    ok       = [r for r in results if r.status == "ok"]
    failed   = [r for r in results if r.status == "failed"]
    skipped  = [r for r in results if r.status == "skipped"]

    summary = {
        "experiment":  name,
        "description": desc,
        "date":        date,
        "run_id":      RUN_ID,
        "total":       len(combos),
        "ok":          len(ok),
        "failed":      len(failed),
        "skipped":     len(skipped),
        "elapsed_s":   elapsed,
        "results": [
            {
                "task":         r.combo.task,
                "model":        r.combo.model,
                "system":       r.combo.system,
                "status":       r.status,
                "overall_mean": r.overall_mean,
                "elapsed_s":    r.elapsed,
                "output":       r.output,
                "error":        r.error or None,
            }
            for r in sorted(results, key=lambda r: (r.combo.task, r.combo.model))
        ],
    }

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _log.info(
        "=== '%s' finished: %d ok  %d failed  %d skipped  %.1fs ===",
        name, len(ok), len(failed), len(skipped), elapsed,
    )
    _log.info("Summary → %s", summary_path)

    if failed:
        _log.warning("Failed combos:")
        for r in failed:
            _log.warning("  %s / %s / %s: %s", r.combo.task, r.combo.model, r.combo.system, r.error)

    return summary


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or all experiments defined in configs/experiments.yaml"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experiment", "-e", metavar="NAME",
                       help="Run a single named experiment")
    group.add_argument("--all", action="store_true",
                       help="Run every experiment in experiments.yaml")
    group.add_argument("--list", "-l", action="store_true",
                       help="List all experiments and their combo counts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print combos without running; show which already exist")
    args = parser.parse_args()

    experiments = _load_experiments()

    if args.list:
        print(f"\nExperiments in {_EXPERIMENTS_YAML.name}:\n")
        for name, cfg in experiments.items():
            combos = _expand(cfg)
            print(f"  {name:<25} {len(combos):>3} combos  "
                  f"parallel={cfg.get('parallel', True)}  "
                  f"workers={cfg.get('max_workers', 3)}")
            print(f"    {cfg.get('description', '')}")
        print()
        return

    if args.list or (not args.experiment and not args.all):
        parser.print_help()
        return

    targets = experiments if args.all else {args.experiment: experiments[args.experiment]}

    if args.experiment and args.experiment not in experiments:
        _log.error("Experiment '%s' not found. Run --list to see available.", args.experiment)
        raise SystemExit(1)

    overall_t0 = time.time()
    all_summaries = {}
    for name, cfg in targets.items():
        all_summaries[name] = run_experiment(name, cfg, dry_run=args.dry_run)

    if len(targets) > 1 and not args.dry_run:
        total_elapsed = round(time.time() - overall_t0, 1)
        total_ok = sum(s.get("ok", 0) for s in all_summaries.values())
        total_fail = sum(s.get("failed", 0) for s in all_summaries.values())
        _log.info("=== All experiments done: %d ok  %d failed  %.1fs total ===",
                  total_ok, total_fail, total_elapsed)


if __name__ == "__main__":
    main()
