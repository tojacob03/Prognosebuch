# Contributing

Thanks for your interest. Issues and pull requests are welcome, in English or German.

## Ground rules of the book

These rules keep the track record trustworthy and are enforced by CI:

1. **Files under `forecasts/` are never modified or deleted**, not even to fix a bug. A wrong
   forecast stays in the book; the fix goes into a new model version.
2. **Changing a model's logic means a new version** (`version="2"` in its definition, a new
   entry in `src/prognosebuch/registry.py` with `live_since` set to a future date). The old
   version keeps running or gets `retired_after`; its record stays.
3. **No information from after the cutoff.** New inputs need a test like
   `tests/test_leakage.py` and evidence (e.g. from the `probe` branch) that the data exists before
   the issue time. Weather training data must come from lead times at least as long as live.
4. **Only openly licensed data.** Add every source to `DATA_SOURCES.md` with its license and
   obligations before using it.
5. Backtest results are never presented as live results.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest            # offline; fixtures in tests/fixtures
uv run prognosebuch audit
```

Adding a model: implement `point_for(prices, issue, target)` plus `error_days` and
`min_error_days` (see `models/naive.py`, `models/lear.py`), use `predict_with_bands`, add it to
`CATALOG`, run `prognosebuch backtest`, and only then register it live.

Code: MIT. By contributing you agree that your contributions are licensed under MIT (code) and
CC BY 4.0 (data).
