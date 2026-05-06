# auto_benchmark_cl

Continual learning benchmark: measures how LLM-based systems retain and transfer knowledge across sequential tasks.

## Architecture
- `src/systems/` — pluggable agent systems (stateless, ICL, SAGE)
- `src/tasks/` — thin wrappers around CL-Bench tasks
- `src/utils/llm.py` — ALL LLM calls go through LiteLLM; never call openai/anthropic SDK directly
- `configs/` — model strings and experiment schedules

## Active tasks
- `codebase_adaptation` — adapt to evolving codebases across API/refactor shifts
- `sales_prediction` — predict sales under distribution shift

## Systems
- **Stateless**: pure prompt → response, no memory
- **ICL**: in-context examples carried in the prompt window
- **SAGE**: neuromodulated episodic memory (DA/NE/ACh/5-HT) — scaffold ready, implementation TBD

## Running experiments
```bash
conda activate twopac
cp .env.example .env  # fill in your keys
bash scripts/run_horse_race.sh   # Phase 1: compare LLMs
bash scripts/run_sage.sh          # Phase 2: SAGE system
```

## Keys
`OPENAI_API_KEY` and `OPENROUTER_API_KEY` required. See `.env.example`.

## Metrics
Gain, BWT (backward transfer), FWT (forward transfer) — `src/utils/metrics.py`.

## Constraints
- Python 3.12, conda env `tropic`
- OpenRouter model strings: `openrouter/<provider>/<model>`
