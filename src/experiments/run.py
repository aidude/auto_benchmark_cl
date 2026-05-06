"""
Run one (system × model × task) combo and write JSON results.

Results land at:  results/YYYY-MM-DD/<run_id>/<task>_<model>_<system>.json
Logs land at:     logs/YYYY-MM-DD/run_<run_id>.log
                  logs/YYYY-MM-DD/llm_calls_<run_id>.jsonl

Usage:
  python -m src.experiments.run \\
    --task  sales_prediction \\
    --model gpt-4o-mini \\
    --system stateless \\
    --output results/2026-05-07/run_143022/out.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import get_logger, RUN_ID
from src.systems.stateless import StatelessSystem
from src.systems.icl import ICLSystem
from src.tasks.sales_prediction import SalesPredictionTask

_log = get_logger(__name__)

SYSTEMS = {
    "stateless": StatelessSystem,
    "icl":       ICLSystem,
}

TASKS = {
    "sales_prediction": SalesPredictionTask,
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def default_output_path(task: str, model: str, system: str) -> Path:
    """Daywise default: results/YYYY-MM-DD/<RUN_ID>/<task>_<model_slug>_<system>.json"""
    date = datetime.now().strftime("%Y-%m-%d")
    slug = model.replace("/", "-")
    return _PROJECT_ROOT / "results" / date / RUN_ID / f"{task}_{slug}_{system}.json"


def run(system_name: str, model: str, task_name: str) -> dict:
    _log.info("START  system=%s  model=%s  task=%s", system_name, model, task_name)
    task = TASKS[task_name]()
    system = SYSTEMS[system_name](model=model)

    per_task: list[dict] = []
    t0 = time.time()

    for task_id in range(task.num_tasks):
        scores: list[float] = []
        episode_log: list[dict] = []

        for ep in task.iter_episodes(task_id):
            obs = {"text": ep.text, "task_id": task_id}
            action = system.act(obs)
            reward = task.score(action, ep.target)
            system.update(obs, action, reward)
            scores.append(reward)
            episode_log.append({
                "target":   ep.target,
                "response": action,
                "score":    round(reward, 4),
            })

        mean = sum(scores) / len(scores)
        _log.info("  Task %d (%s): mean_score=%.4f  episodes=%d", task_id, ep.regime, mean, len(scores))
        per_task.append({
            "task_id":    task_id,
            "regime":     ep.regime,
            "mean_score": round(mean, 4),
            "episodes":   episode_log,
        })

    overall = round(sum(t["mean_score"] for t in per_task) / len(per_task), 4)
    elapsed = round(time.time() - t0, 1)
    _log.info("DONE   overall_mean=%.4f  elapsed=%.1fs", overall, elapsed)

    return {
        "run_id":       RUN_ID,
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "system":       system_name,
        "model":        model,
        "task":         task_name,
        "per_task":     per_task,
        "overall_mean": overall,
        "elapsed_s":    elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   required=True, choices=list(TASKS))
    parser.add_argument("--model",  required=True)
    parser.add_argument("--system", default="stateless", choices=list(SYSTEMS))
    parser.add_argument("--output", default=None,
                        help="Override output path; default: daywise results dir")
    args = parser.parse_args()

    result = run(args.system, args.model, args.task)

    out = Path(args.output) if args.output else default_output_path(
        args.task, args.model, args.system
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
