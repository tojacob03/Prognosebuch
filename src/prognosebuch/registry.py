"""Models that run live and count for the public book.

Rules:
- A model is identified by (name, version). Changing a model's logic means a new version,
  never an edit of an existing one; old versions keep running or are retired, but their
  record stays in the book.
- ``live_since`` is the first issue date (Europe/Berlin) from which a missing forecast
  counts as a missed day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from prognosebuch.models.bands import PointModel
from prognosebuch.models.gbm import GBM_V1
from prognosebuch.models.lear import LEAR_V1
from prognosebuch.models.naive import NAIVE_LAST_DAY, NAIVE_SIMILAR_DAY, NAIVE_WEEKLY

HORIZONS: tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class LiveModel:
    model: PointModel
    live_since: date
    retired_after: date | None = None

    @property
    def key(self) -> str:
        return f"{self.model.name}.v{self.model.version}"

    def expected_on(self, issue_date: date) -> bool:
        if issue_date < self.live_since:
            return False
        return self.retired_after is None or issue_date <= self.retired_after


LIVE_MODELS: tuple[LiveModel, ...] = (
    LiveModel(NAIVE_LAST_DAY, live_since=date(2026, 9, 30)),
    LiveModel(NAIVE_WEEKLY, live_since=date(2026, 9, 30)),
    LiveModel(NAIVE_SIMILAR_DAY, live_since=date(2026, 9, 30)),
    LiveModel(LEAR_V1, live_since=date(2026, 9, 30)),
    LiveModel(GBM_V1, live_since=date(2026, 9, 30)),
)

# Skill scores are reported relative to this model.
REFERENCE_MODEL_KEY = "naive_similar_day.v1"

# Every model version that exists (live or not), for backtests.
CATALOG: dict[str, PointModel] = {
    f"{m.name}.v{m.version}": m
    for m in (NAIVE_LAST_DAY, NAIVE_WEEKLY, NAIVE_SIMILAR_DAY, LEAR_V1, GBM_V1)
}
