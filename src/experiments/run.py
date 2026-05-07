"""
Run one (system × model × task) combo and write JSON results.

Results: results/YYYY-MM-DD/<run_id>/<task>_<model>_<system>.json
Logs:    logs/YYYY-MM-DD/run_<run_id>.log
         logs/YYYY-MM-DD/llm_calls_<run_id>.jsonl

Usage:
  python -m src.experiments.run --task sales_prediction --model gpt-4o-mini
"""
import argparse
import json
import os
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from src.utils.logger import get_logger, RUN_ID
from src.systems.stateless import StatelessSystem
from src.systems.icl import ICLSystem
from src.tasks import SalesPredictionTask

_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_YAML  = _PROJECT_ROOT / "configs" / "models.yaml"

_MAX_WORKERS = 10  # concurrent LLM calls for parallel_safe systems


# ── Model registry ───────────────────────────────────────────

@dataclass
class ModelInfo:
    short_name:     str
    litellm_id:     str
    provider:       str
    model_type:     str   # "reasoning" | "non_reasoning"
    endpoint:       str
    context_window: int
    requires_key:   str | None
    params:         dict


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, ModelInfo]:
    """Parse configs/models.yaml once and cache for the process lifetime."""
    cfg = yaml.safe_load(_MODELS_YAML.read_text())
    registry: dict[str, ModelInfo] = {}

    for provider, type_groups in cfg.items():
        if provider == "horse_race" or not isinstance(type_groups, dict):
            continue
        for model_type, models in type_groups.items():
            if not isinstance(models, dict):
                continue
            for short_name, spec in models.items():
                if not isinstance(spec, dict):
                    continue
                params = dict(spec.get("params") or {})
                if "budget_tokens" in params:
                    budget = params.pop("budget_tokens")
                    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
                registry[short_name] = ModelInfo(
                    short_name=short_name,
                    litellm_id=spec["litellm"],
                    provider=provider,
                    model_type=model_type,
                    endpoint=spec.get("endpoint", ""),
                    context_window=spec.get("context_window", 0),
                    requires_key=spec.get("requires_key"),
                    params=params,
                )
    return registry


def resolve_model(name: str) -> ModelInfo:
    registry = _load_registry()
    if name in registry:
        info = registry[name]
        _log.info(
            "Model: %s → %s  [%s / %s]  params=%s",
            name, info.litellm_id, info.provider, info.model_type, info.params or "none",
        )
        return info
    _log.warning("'%s' not in registry — using as raw litellm id", name)
    return ModelInfo(
        short_name=name, litellm_id=name, provider="unknown",
        model_type="non_reasoning", endpoint="", context_window=0,
        requires_key=None, params={},
    )


def _check_key(info: ModelInfo) -> None:
    if info.requires_key and not os.getenv(info.requires_key):
        raise EnvironmentError(
            f"Model '{info.short_name}' requires {info.requires_key} — set it in .env"
        )


# ── Systems / Tasks ───────────────────────────────────────────

SYSTEMS: dict[str, type] = {
    "stateless": StatelessSystem,
    "icl":       ICLSystem,
}

TASKS: dict[str, type] = {
    "sales_prediction": SalesPredictionTask,
}

# Steers reasoning models away from verbose self-correction loops.
# Targets "Wait / Hmm / let me reconsider" patterns that balloon token counts.
_REASONING_SYSTEM_PROMPT = (
    "You are a precise forecasting assistant. "
    "Think briefly, then give your final answer immediately. "
    "Keep reasoning to 3 steps or fewer. "
    "Do not use filler words like 'wait' or 'hmm', and do not self-correct. "
    "Your response must be a single integer."
)


# ── Episode runner ────────────────────────────────────────────

def _run_task(system, task, task_id: int) -> tuple[list[float], list[dict], str]:
    """Run all episodes for one task.

    Parallel systems (parallel_safe=True): all act() calls fire concurrently,
    then update() is called in order after all futures complete.

    Serial systems (ICL, SAGE, …): act → score → update are interleaved per
    episode so the in-context buffer is current for every subsequent call.
    """
    episodes = list(task.iter_episodes(task_id))
    regime = episodes[0].regime
    scored: list[tuple[str, float]] = []

    bar = tqdm(
        total=len(episodes),
        desc=f"    {regime:<15}",
        unit="ep",
        leave=False,
        dynamic_ncols=True,
    )

    if getattr(system, "parallel_safe", False):
        # Fire all LLM calls concurrently; collect in submission order.
        indexed: list[tuple[str, float]] = [None] * len(episodes)

        def _call(idx: int, ep):
            action = system.act({"text": ep.text, "task_id": task_id})
            reward = task.score(action, ep.target)
            return idx, action, reward

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_call, i, ep): i for i, ep in enumerate(episodes)}
            for fut in as_completed(futures):
                i, action, reward = fut.result()
                indexed[i] = (action, reward)
                bar.update(1)

        # update() in submission order after all parallel calls complete
        for ep, (action, reward) in zip(episodes, indexed):
            system.update({"text": ep.text, "task_id": task_id}, action, reward)
        scored = indexed
    else:
        # Serial: interleave act → score → update so each episode sees the
        # accumulated in-context buffer from all previous episodes.
        for ep in episodes:
            obs = {"text": ep.text, "task_id": task_id}
            action = system.act(obs)
            reward = task.score(action, ep.target)
            system.update(obs, action, reward)
            scored.append((action, reward))
            bar.update(1)

    bar.close()

    scores, episode_log = [], []
    for ep, (action, reward) in zip(episodes, scored):
        scores.append(reward)
        episode_log.append({"target": ep.target, "response": action, "score": round(reward, 4)})

    return scores, episode_log, regime


# ── Experiment entry point ────────────────────────────────────

def run(system_name: str, info: ModelInfo, task_name: str,
        outpath: Path | None = None) -> dict:
    _log.info("START  system=%s  model=%s  type=%s  task=%s",
              system_name, info.litellm_id, info.model_type, task_name)
    task = TASKS[task_name]()
    system_prompt = _REASONING_SYSTEM_PROMPT if info.model_type == "reasoning" else ""
    if system_prompt:
        _log.info("  reasoning prompt active — brevity steering enabled")
    system = SYSTEMS[system_name](model=info.litellm_id, system_prompt=system_prompt, llm_kwargs=info.params)

    per_task, t0 = [], time.time()
    num_tasks = task.num_tasks

    with tqdm(total=num_tasks, desc=f"  {info.short_name}/{system_name}", unit="task",
              dynamic_ncols=True) as task_bar:
        for task_id in range(num_tasks):
            scores, episode_log, regime = _run_task(system, task, task_id)
            mean = sum(scores) / len(scores)
            _log.info("  Task %d (%s): mean=%.4f  n=%d", task_id, regime, mean, len(scores))
            per_task.append({"task_id": task_id, "regime": regime,
                             "mean_score": round(mean, 4), "episodes": episode_log})

            task_bar.set_postfix(regime=regime, mean=f"{mean:.3f}")
            task_bar.update(1)

            # Write partial result after each task so progress is visible mid-run
            if outpath:
                _write_result(outpath, system_name, info, task_name,
                              per_task, t0, complete=False)

    overall = round(sum(t["mean_score"] for t in per_task) / len(per_task), 4)
    elapsed = round(time.time() - t0, 1)
    _log.info("DONE   overall=%.4f  elapsed=%.1fs", overall, elapsed)

    result = {
        "run_id":       RUN_ID,
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "system":       system_name,
        "model":        info.litellm_id,
        "model_short":  info.short_name,
        "provider":     info.provider,
        "model_type":   info.model_type,
        "task":         task_name,
        "per_task":     per_task,
        "overall_mean": overall,
        "elapsed_s":    elapsed,
        "complete":     True,
    }
    if outpath:
        outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_result(outpath: Path, system_name: str, info: ModelInfo,
                  task_name: str, per_task: list, t0: float, complete: bool) -> None:
    """Write partial or final result JSON to outpath."""
    partial_mean = round(sum(t["mean_score"] for t in per_task) / len(per_task), 4)
    outpath.write_text(json.dumps({
        "run_id":       RUN_ID,
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "system":       system_name,
        "model":        info.litellm_id,
        "model_short":  info.short_name,
        "provider":     info.provider,
        "model_type":   info.model_type,
        "task":         task_name,
        "per_task":     per_task,
        "overall_mean": partial_mean,
        "elapsed_s":    round(time.time() - t0, 1),
        "complete":     complete,
    }, indent=2), encoding="utf-8")


def _default_output(task: str, model_short: str, system: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = model_short.replace("/", "-")
    return _PROJECT_ROOT / "results" / date / RUN_ID / f"{task}_{slug}_{system}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   required=True, choices=list(TASKS))
    parser.add_argument("--model",  required=True,
                        help="Short name from configs/models.yaml, e.g. gpt-4o-mini")
    parser.add_argument("--system", default="stateless", choices=list(SYSTEMS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    info = resolve_model(args.model)
    _check_key(info)

    out = Path(args.output) if args.output else _default_output(args.task, args.model, args.system)
    out.parent.mkdir(parents=True, exist_ok=True)
    run(args.system, info, args.task, outpath=out)
    _log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
