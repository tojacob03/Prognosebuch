"""Build the whole website from a small book and check pages and exports."""

import json
import re
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from prognosebuch.evaluate import run_evaluate
from prognosebuch.forecast import RunInfo, run_forecast
from prognosebuch.site.build import build_site
from prognosebuch.site.i18n import PAGES, num, pct
from prognosebuch.timeutil import day_bounds_utc
from tests.conftest import FakeClient, fast_live_models, seeded_client, synthetic_prices

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("book")
    shutil.copy(ROOT / "METHODOLOGY.md", root / "METHODOLOGY.md")
    models = fast_live_models(live_since=date(2026, 10, 5))
    prices = synthetic_prices(date(2026, 1, 1), date(2026, 10, 9), seed=2)
    for d in (date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)):
        known = prices[prices.index < day_bounds_utc(d)[1]]
        now = datetime(d.year, d.month, d.day, 7, 5, tzinfo=UTC)
        run_forecast(root, now, seeded_client(root, known), RunInfo(None, None), models)
    visible = prices[prices.index < day_bounds_utc(date(2026, 10, 8))[1]]
    run_evaluate(
        root,
        datetime(2026, 10, 7, 14, 0, tzinfo=UTC),
        FakeClient(visible),
        models,
        update_weather_archive=False,
    )
    out = root / "_site"
    build_site(root, out, datetime(2026, 10, 7, 15, 0, tzinfo=UTC))
    return out


def test_all_pages_exist_in_both_languages(site: Path) -> None:
    for paths in PAGES.values():
        for path in paths.values():
            html = (site / path / "index.html").read_text()
            assert "<html lang=" in html
            assert not re.search(r"\b(None|nan|NaN)\b", html), path


def test_home_shows_latest_forecast_and_book(site: Path) -> None:
    de = (site / "index.html").read_text()
    assert "Donnerstag, 8. Oktober 2026" in de
    assert "Günstigste 3 Stunden" in de and "<svg" in de
    en = (site / "en" / "index.html").read_text()
    assert "Thursday, 8 October 2026" in en


def test_exports(site: Path) -> None:
    data = site / "data"
    latest = json.loads((data / "latest.json").read_text())
    assert latest["issue_date"] == "2026-10-07"
    assert latest["license"] == "CC BY 4.0"
    fc = latest["forecasts"][latest["featured_model"]]
    assert set(fc["targets"]) == {"2026-10-08", "2026-10-09"}
    assert len(fc["targets"]["2026-10-08"]["quarter_hours"]) == 96
    df = pd.read_csv(data / "forecasts.csv")
    assert {"delivery_date_local", "delivery_time_local", "q50"} <= set(df.columns)
    assert df["issued_at_utc"].str.endswith("Z").all()
    assert (data / "data_dictionary.csv").exists() and (data / "scores.csv").exists()
    assert (site / "static" / "fonts" / "source-serif-4.woff2").exists()


def test_number_formatting() -> None:
    assert num(-1234.5, 1, "de") == "\u22121\u2009234,5"
    assert num(1234.5, 1, "en") == "1,234.5"
    assert pct(0.305, "de") == "30\u202f%"
