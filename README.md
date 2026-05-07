# auto_benchmark_cl

Continual learning benchmark that measures how LLM-based agent systems retain and transfer knowledge across sequential tasks.

---

## Design decisions

### Why continual learning over standard benchmarks

Standard LLM evals test a model on a fixed distribution. Real-world deployments face distribution shift — sales patterns change, codebases evolve, user behaviour drifts. This benchmark runs models through sequential tasks where the underlying data distribution changes across regimes (e.g. `stable_growth → holiday_spike → market_dip → recovery`), measuring not just peak performance but retention (BWT) and forward transfer (FWT).

### Pluggable system architecture

Agent behaviour is separated from task execution via `BaseSystem` in `src/systems/`. Each system implements `act / update / reset`. This lets us swap memory strategies — stateless, ICL, SAGE — against the same task without touching task code. The task only sees observations and scores actions; it has no knowledge of what system is running.

### Why LiteLLM as the single call layer

All LLM calls go through `src/utils/llm.complete()` and never call provider SDKs directly. This gives a single place to enforce retry policy, timeout tiers, audit logging, and cost tracking regardless of provider. Adding a new model means editing `configs/models.yaml`, not touching any system or task code.

### Retry and timeout strategy

Transient failures (rate limits, 5xx, connection drops) are retried up to 3 times with exponential backoff (base 2s, doubles each attempt, +random jitter). Timeouts are tiered by model size in `models.yaml` — small reasoning models get 30s, medium 60s, large 120s — rather than a single global cap that would either time out large models too early or leave hung small-model calls blocking indefinitely.

### ICL serial execution with interleaved update

Stateless systems fire all episode calls concurrently (`parallel_safe=True`, `ThreadPoolExecutor`). ICL and SAGE are serial (`parallel_safe=False`) with `act → score → update` interleaved per episode. The reason: if all `act()` calls fire first and `update()` runs after, the in-context buffer is empty for every call within a task, defeating the purpose of ICL entirely. Serial interleaving ensures each episode sees the accumulated examples from all prior episodes in the same task.

### ICL buffer sizing and per-task reset

`max_examples=5` (down from 10). Sales prediction observations are ~1,200–1,800 tokens each. At 10 examples the buffer pushed input tokens to 14k–18k per call, exceeding context limits on smaller models and burning disproportionate credits on OpenRouter. Five examples keeps the buffer under ~9k tokens while still providing meaningful in-context history.

The buffer is reset between tasks (`system.reset()` in `run.py` after each task completes). Without this, examples from task N carry into task N+1, so tasks 1–3 start already at max capacity. Resetting keeps token counts predictable and ensures each task's ICL window only contains examples from that task's own distribution.

### Reasoning trace stripping

Reasoning models (qwen3, deepseek-r1, etc.) embed chain-of-thought in `<think>...</think>` blocks inside the response content. Without stripping, ICL stores the full trace as the `action` — up to ~3k tokens per example, making the buffer ~15k tokens of noise even at `max_examples=5`. `llm.py` strips these blocks before returning, so stored actions are just the final answer. If stripping produces an empty string (some models embed the answer inside the block), the code falls back to the post-`</think>` content, then to the raw text.

### max_tokens caps for reasoning models

Sales prediction asks for a single integer. Reasoning models do not need 4,000 output tokens for that. Small and medium reasoning models are capped at `max_tokens=1500` in `models.yaml`. Large models (o1, o3, gpt-5.4, deepseek-r1, deepseek-v4-Pro, claude-opus-4-7-think) keep 4,000 since they are used sparingly and genuinely benefit from deeper thinking budgets. For Anthropic extended-thinking models `budget_tokens` is set below `max_tokens` (1,200 / 1,500) to leave headroom for the actual response.

### Brevity steering for reasoning models

A system prompt is injected for all `model_type == "reasoning"` models that discourages self-correction loops (`wait / hmm / let me reconsider` patterns). These patterns balloon token counts without improving accuracy on structured prediction tasks. Non-reasoning models receive no system prompt injection.

### Incremental result writes and implicit resume

`run()` writes a partial JSON file (`complete: false`) after every task and overwrites it with `complete: true` on finish. The batch runner checks this flag on startup — completed combos are skipped automatically, so re-running an interrupted experiment resumes from where it left off without any explicit checkpoint management.

### Parallelism levels

| Level | Strategy | Reason |
|---|---|---|
| Experiments | Serial (intentional) | Avoid cross-experiment rate-limit contention |
| Combos within an experiment | `ThreadPoolExecutor` (configurable `max_workers`) | Independent runs, no shared state |
| Episodes within a task (stateless) | `ThreadPoolExecutor` (10 workers) | All calls are independent |
| Episodes within a task (ICL/SAGE) | Serial | Update must precede next act |

### Model registry as the single source of truth

`configs/models.yaml` owns all model configuration — litellm ID, endpoint, context window, required API key, and extra params (`max_tokens`, `timeout`, `budget_tokens`). Systems and tasks receive a `ModelInfo` dataclass resolved at startup. This means changing a model's timeout or token cap requires editing one file, not hunting through system code.

---

## Task suite

All six tasks from the [upstream CL benchmark](https://continual-learning-bench.com) are implemented as text-in/text-out wrappers following our `iter_episodes / score / num_tasks` interface. Each task has 4 sequential regimes × 30 episodes. The regimes simulate the distribution shift that makes continual learning hard.

| Task slug | Domain | Regimes (distribution shift) | Output format | Scoring |
|---|---|---|---|---|
| `sales_prediction` | Time-series forecasting | stable_growth → holiday_spike → market_dip → recovery | Single integer | Relative error ≤ 100% |
| `exploitable_poker` | Game theory / strategy | calling_station → random_folder → bluff_heavy → tight_aggressive | FOLD / CALL / CHECK / RAISE X | Action type match + raise amount accuracy |
| `database_exploration` | Data querying | schema_v1 → column_drift → table_split → schema_merge | Single integer | Relative error ≤ 10% |
| `cohort_studies` | Clinical / epidemiology | oncology_early → oncology_late → cardiology_acute → cardiology_chronic | Survival probability [0.00–1.00] | Absolute error tolerance |
| `blind_spectrum_monitoring` | Signal processing / RF | clean_band → interference → sensor_drift → reconfigured | `<signal_type>:<freq_MHz>` | Type exact + freq ±5 MHz |
| `codebase_adaptation` | Software engineering | api_v1 → api_v2_rename → api_v3_params → api_v4_breaking | Corrected function call string | Exact match / method name match |

Pass any slug to `--task` or reference it in `experiments.yaml`. The upstream benchmark uses Docker containers and live execution for some tasks; our wrappers are synthetic but capture the same CL challenge (learnable patterns within each regime, abrupt shift across regimes).

---

## Metrics

| Metric | Description |
|---|---|
| Gain | Mean score across all episodes and tasks |
| BWT (backward transfer) | How much learning on later tasks degrades performance on earlier tasks |
| FWT (forward transfer) | How much earlier tasks improve performance on later unseen tasks |
