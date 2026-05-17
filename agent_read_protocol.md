# Agent code-map reading protocol

1. Read the relevant `*_code_map.json` first.
2. Check `file_summary`, then `symbol_map`, then `main_phases`.
3. Pick only the smallest relevant symbol or phase.
4. Read that code section next.
5. Append the visited symbol/line range to agent state.
6. Never re-read a visited section unless a new question depends on it.
7. For `runner.py`, separate A1 and A2 questions early.
8. For `infer.py`, decide first whether the question is about config, calibration, model rebuild, or CSV export.

## Suggested state
```json
{
  "visited_symbols": [],
  "visited_line_ranges": [],
  "selected_task_mode": null,
  "open_questions": []
}
```

## Dispatch rule
- If the question is about training, validation, calibration, metrics, or submission generation during training → `runner.py`
- If the question is about release inference, checkpoint restore, calibration loading, or final CSV output → `infer.py`
