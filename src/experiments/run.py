"""
Run one (system × model × task) combo and write JSON results.

Usage:
  python -m src.experiments.run \\
    --task  sales_prediction \\
    --model gpt-4o-mini \\
    --system stateless \\
    --output results/run.json
"""
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.systems.stateless import StatelessSystem
from src.systems.icl import ICLSystem
from src.tasks.sales_prediction import SalesPredictionTask

SYSTEMS = {
    "stateless": StatelessSystem,
    "icl": ICLSystem,
}

TASKS = {
    "sales_prediction": SalesPredictionTask,
}


def run(system_name: str, model: str, task_name: str) -> dict:
    task = TASKS[task_name]()
    system = SYSTEMS[system_name](model=model)

    per_task: list[dict] = []

    for task_id in range(task.num_tasks):
        scores, responses = [], []
        for ep in task.iter_episodes(task_id):
            obs = {"text": ep.text, "task_id": task_id}
            action = system.act(obs)
            reward = task.score(action, ep.target)
            system.update(obs, action, reward)
            scores.append(reward)
            responses.append({"target": ep.target, "response": action, "score": reward})

        mean = sum(scores) / len(scores)
        print(f"  Task {task_id} ({ep.regime}): mean_score={mean:.3f}")
        per_task.append({"task_id": task_id, "regime": ep.regime, "mean_score": round(mean, 4)})

    return {
        "system": system_name,
        "model": model,
        "task": task_name,
        "per_task": per_task,
        "overall_mean": round(sum(t["mean_score"] for t in per_task) / len(per_task), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   required=True, choices=list(TASKS))
    parser.add_argument("--model",  required=True)
    parser.add_argument("--system", default="stateless", choices=list(SYSTEMS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"\n=== system={args.system}  model={args.model}  task={args.task} ===")
    t0 = time.time()
    result = run(args.system, args.model, args.task)
    result["elapsed_s"] = round(time.time() - t0, 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out}  (overall_mean={result['overall_mean']})\n")


if __name__ == "__main__":
    main()
