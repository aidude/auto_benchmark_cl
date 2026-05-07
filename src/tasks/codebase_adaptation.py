"""
Codebase Adaptation CL Task.

Four sequential regimes simulating API evolution across a shared library:
  0 — api_v1           (original stable API)
  1 — api_v2_rename    (methods renamed; old names still work but deprecated)
  2 — api_v3_params    (parameter order changed; keyword args required)
  3 — api_v4_breaking  (old method removed entirely; new interface mandatory)

Each regime = 30 issues.
Each episode: LLM sees a code snippet using the old/broken API and a GitHub
              issue describing the failure → outputs the corrected function call.

Target: the corrected function call string (e.g. "client.send_request(url, method='POST')")
Score: 1.0 exact match; 0.5 if method name correct but args differ; 0.0 otherwise.

Note: this is a simplified text-only version of the upstream Docker-based task.
"""
import random
import re
from dataclasses import dataclass
from typing import Iterator

EPISODES_PER_REGIME = 30

# Each entry: (regime, old_call_template, new_call_template, issue_template)
_REGIMES = [
    (
        "api_v1",
        "client.{method}({args})",
        "client.{method}({args})",
        "Confirm the correct call for {method} with {args}.",
    ),
    (
        "api_v2_rename",
        "client.{old_method}({args})",
        "client.{new_method}({args})",
        "DeprecationWarning: '{old_method}' is deprecated. Use '{new_method}' instead.",
    ),
    (
        "api_v3_params",
        "client.{method}({positional_args})",
        "client.{method}({keyword_args})",
        "TypeError: positional arguments are no longer supported; use keyword arguments.",
    ),
    (
        "api_v4_breaking",
        "client.{old_method}({args})",
        "client.{new_method}({new_args})",
        "AttributeError: '{old_method}' has been removed in v4. See migration guide.",
    ),
]

_METHOD_PAIRS = [
    ("fetch_data",      "get_data",       "retrieve_data"),
    ("send_request",    "post_request",   "submit_request"),
    ("parse_response",  "decode_response","read_response"),
    ("connect",         "open_connection","establish_connection"),
    ("upload_file",     "push_file",      "write_file"),
    ("list_records",    "get_records",    "fetch_records"),
    ("delete_item",     "remove_item",    "drop_item"),
    ("update_config",   "set_config",     "apply_config"),
]

_ARG_SETS = [
    ('"https://api.example.com/v1"', '"https://api.example.com/v1"',
     'url="https://api.example.com/v1"'),
    ('"records", limit=10',          '"records", limit=10',
     'resource="records", limit=10'),
    ('"user_123"',                    '"user_123"',
     'id="user_123"'),
    ('"config.yaml", overwrite=True', '"config.yaml", overwrite=True',
     'path="config.yaml", overwrite=True'),
]


@dataclass
class Episode:
    task_id: int
    regime: str
    target: str
    text: str


def _format_prompt(regime: str, broken_code: str, issue: str, context: str) -> str:
    return (
        "You are fixing a bug in a Python codebase.\n\n"
        f"GitHub issue  : {issue}\n\n"
        f"Broken code   :\n  {broken_code}\n\n"
        f"API context   : {context}\n\n"
        "Respond with only the corrected single-line function call. "
        "No explanation, no markdown, no import statements."
    )


def _build_episode(rng: random.Random, task_id: int, regime_label: str,
                   regime_idx: int) -> Episode:
    v1_name, v2_name, v4_name = rng.choice(_METHOD_PAIRS)
    pos_args, kw_same, kw_args = rng.choice(_ARG_SETS)

    if regime_idx == 0:   # api_v1 — confirm the current correct call
        old_call = f"client.{v1_name}({pos_args})"
        new_call = old_call
        issue    = f"Confirm the correct call for {v1_name}."
        context  = f"Use client.{v1_name}(). No changes in v1."

    elif regime_idx == 1:  # api_v2 rename
        old_call = f"client.{v1_name}({pos_args})"
        new_call = f"client.{v2_name}({pos_args})"
        issue    = f"DeprecationWarning: '{v1_name}' is deprecated. Use '{v2_name}' instead."
        context  = f"Renamed: {v1_name} → {v2_name}. Arguments unchanged."

    elif regime_idx == 2:  # api_v3 keyword args
        old_call = f"client.{v2_name}({pos_args})"
        new_call = f"client.{v2_name}({kw_args})"
        issue    = (f"TypeError: {v2_name}() no longer accepts positional arguments. "
                    "Use keyword arguments.")
        context  = f"v3: {v2_name}() requires all arguments as keyword arguments."

    else:                  # api_v4 breaking rename
        old_call = f"client.{v2_name}({kw_args})"
        new_call = f"client.{v4_name}({kw_args})"
        issue    = f"AttributeError: '{v2_name}' removed in v4. Migration: use '{v4_name}'."
        context  = f"Breaking change: {v2_name} removed. Replacement: {v4_name}() same signature."

    text = _format_prompt(regime_label, old_call, issue, context)
    return Episode(task_id=task_id, regime=regime_label, target=new_call, text=text)


def _parse(response: str) -> str:
    return response.strip().split("\n")[0].strip()


def _method_name(call: str) -> str:
    m = re.search(r"\.(\w+)\(", call)
    return m.group(1) if m else ""


class CodebaseAdaptationTask:
    """Four-regime CL task with progressive API breaking changes."""

    def __init__(self, seed: int = 42):
        self._regimes = self._build(seed)

    def _build(self, seed: int) -> list[list[Episode]]:
        regimes = []
        for task_id, (regime_label, *_) in enumerate(_REGIMES):
            rng = random.Random(seed + task_id * 7)
            episodes = [
                _build_episode(rng, task_id, regime_label, task_id)
                for _ in range(EPISODES_PER_REGIME)
            ]
            regimes.append(episodes)
        return regimes

    @property
    def num_tasks(self) -> int:
        return len(self._regimes)

    def iter_episodes(self, task_id: int) -> Iterator[Episode]:
        yield from self._regimes[task_id]

    @staticmethod
    def score(response: str, target: str) -> float:
        """1.0 exact match; 0.5 method name correct but args differ; 0.0 otherwise."""
        pred = _parse(response)
        if pred == target:
            return 1.0
        if _method_name(pred) == _method_name(target):
            return 0.5
        return 0.0
