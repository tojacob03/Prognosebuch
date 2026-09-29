"""Site texts in German and English, and locale-aware number and date formatting."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

LANGS = ("de", "en")

WEEKDAYS = {
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
MONTHS = {
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
           "September", "Oktober", "November", "Dezember"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
}  # fmt: skip


def num(x: float | None, digits: int = 1, lang: str = "de", sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"{x:+,.{digits}f}" if sign else f"{x:,.{digits}f}"
    s = s.replace("-", "−")
    if lang == "de":
        s = s.replace(",", " ").replace(".", ",")
    return s


def pct(x: float | None, lang: str = "de") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{num(100 * x, 0, lang)} %"


def long_date(d: date, lang: str, weekday: bool = True) -> str:
    wd = WEEKDAYS[lang][d.weekday()]
    if lang == "de":
        s = f"{d.day}. {MONTHS[lang][d.month - 1]} {d.year}"
    else:
        s = f"{d.day} {MONTHS[lang][d.month - 1]} {d.year}"
    return f"{wd}, {s}" if weekday else s


def short_date(d: date, lang: str) -> str:
    wd = WEEKDAYS[lang][d.weekday()][:2]
    if lang == "de":
        return f"{wd} {d.day:02d}.{d.month:02d}."
    return f"{wd} {d.day} {MONTHS[lang][d.month - 1][:3]}"


# Page keys -> path per language (directory URLs).
PAGES: dict[str, dict[str, str]] = {
    "home": {"de": "", "en": "en/"},
    "record": {"de": "bilanz/", "en": "en/track-record/"},
    "failures": {"de": "versagen/", "en": "en/failures/"},
    "cheapest": {"de": "guenstig/", "en": "en/cheapest-hours/"},
    "method": {"de": "methodik/", "en": "en/methodology/"},
    "data": {"de": "daten/", "en": "en/data/"},
    "imprint": {"de": "impressum/", "en": "en/imprint/"},
}

T: dict[str, dict[str, str]] = {
    "site_title": {"de": "Prognosebuch", "en": "Prognosebuch"},
    "tagline": {
        "de": "Strompreis-Prognosen für morgen, vorab festgehalten und öffentlich bewertet",
        "en": "Electricity price forecasts for tomorrow, recorded in advance and scored in public",
    },
    "nav_home": {"de": "Prognose", "en": "Forecast"},
    "nav_record": {"de": "Bilanz", "en": "Track record"},
    "nav_failures": {"de": "Wo es versagt", "en": "Where it fails"},
    "nav_cheapest": {"de": "Günstige Stunden", "en": "Cheapest hours"},
    "nav_method": {"de": "Methodik", "en": "Methodology"},
    "nav_data": {"de": "Daten", "en": "Data"},
    "other_lang": {"de": "English", "en": "Deutsch"},
    "skip": {"de": "Zum Inhalt springen", "en": "Skip to content"},
    # home
    "home_lede": {
        "de": "Day-Ahead-Preis Deutschland-Luxemburg je Viertelstunde. Ausgegeben am {issued}, "
        "vor dem Auktionsschluss um 12:00.",
        "en": "Day-ahead price Germany-Luxembourg per quarter-hour. Issued on {issued}, before "
        "the auction closes at 12:00.",
    },
    "home_empty_title": {"de": "Das Buch beginnt am {date}", "en": "The book opens on {date}"},
    "home_empty": {
        "de": "Die erste Prognose wird am {date} gegen 09:00 Uhr ausgegeben und hier gezeigt. "
        "Bis dahin gibt es nur den Backtest, und der ist kein Nachweis.",
        "en": "The first forecast will be issued on {date} at about 09:00 (Berlin) and shown "
        "here. Until then there is only the backtest, which is not evidence.",
    },
    "tab_d1": {"de": "Morgen", "en": "Tomorrow"},
    "tab_d2": {"de": "Übermorgen", "en": "Day after"},
    "median": {"de": "Median", "en": "Median"},
    "band": {"de": "80-%-Band (P10–P90)", "en": "80 % band (P10–P90)"},
    "actual": {"de": "Tatsächlicher Preis", "en": "Actual price"},
    "cheapest_3h": {"de": "Günstigste 3 Stunden", "en": "Cheapest 3 hours"},
    "day_mean": {"de": "Tagesmittel (Median)", "en": "Day mean (median)"},
    "day_range": {"de": "Viertelstunden von … bis", "en": "Quarter-hours from … to"},
    "model": {"de": "Modell", "en": "Model"},
    "model_rule_ref": {
        "de": "Gezeigt wird die Faustregel, bis ein Modell 14 bewertete Tage hat.",
        "en": "The rule of thumb is shown until a model has 14 scored days.",
    },
    "model_rule_best": {
        "de": "Gezeigt wird das Modell mit dem kleinsten Fehler (D+1) der letzten 30 Tage.",
        "en": "Shown is the model with the lowest D+1 error over the last 30 days.",
    },
    "prob_cheapest": {
        "de": "{p} Wahrscheinlichkeit, dass es das günstigste Fenster ist",
        "en": "{p} probability that this is the cheapest window",
    },
    "prob_near": {
        "de": "{p}, dass es höchstens 0,5 ct/kWh teurer ist als das günstigste",
        "en": "{p} that it costs at most 0.5 ct/kWh more than the cheapest",
    },
    "mae_day": {"de": "Fehler des Tages (MAE)", "en": "Error of the day (MAE)"},
    "book_title": {"de": "Das Buch", "en": "The book"},
    "book_lede": {
        "de": "Jede Zeile ist eine Prognose, die vor der Auktion als unveränderliche Datei "
        "gespeichert wurde. Verpasste Tage bleiben stehen.",
        "en": "Each line is a forecast saved as an immutable file before the auction. Missed "
        "days stay in the book.",
    },
    "delivery_day": {"de": "Liefertag", "en": "Delivery day"},
    "issued_at": {"de": "Ausgegeben", "en": "Issued"},
    "status_pending": {"de": "offen", "en": "pending"},
    "status_missed": {"de": "verpasst", "en": "missed"},
    "check_title": {"de": "Nachprüfen statt glauben", "en": "Check, don't trust"},
    "check_body": {
        "de": "Die Prognosedateien liegen im Git-Repository, jede mit Ausgabezeit, Datenschnitt, "
        "Code-Version und Prüfsumme. Eine automatische Prüfung schlägt fehl, sobald eine "
        "bestehende Datei geändert oder gelöscht wird.",
        "en": "Forecast files live in the git repository, each with issue time, data cutoff, "
        "code version and checksum. An automatic check fails as soon as an existing file is "
        "changed or deleted.",
    },
    "check_links": {"de": "Dateien ansehen", "en": "Browse the files"},
    # record
    "record_title": {"de": "Bilanz", "en": "Track record"},
    "record_lede": {
        "de": "Wie gut waren die Prognosen an echten Tagen, verglichen mit einfachen Faustregeln? "
        "Nur live ausgegebene Prognosen zählen.",
        "en": "How good were the forecasts on real days, compared with simple rules of thumb? "
        "Only forecasts issued live count.",
    },
    "record_empty": {
        "de": "Noch keine bewerteten Tage. Der erste Liefertag ({date}) wird am Nachmittag davor "
        "bewertet, sobald die Preise veröffentlicht sind.",
        "en": "No scored days yet. The first delivery day ({date}) is scored the afternoon "
        "before, once prices are published.",
    },
    "window": {"de": "Zeitraum", "en": "Window"},
    "w_7d": {"de": "7 Tage", "en": "7 days"},
    "w_30d": {"de": "30 Tage", "en": "30 days"},
    "w_90d": {"de": "90 Tage", "en": "90 days"},
    "w_all": {"de": "Gesamt", "en": "All"},
    "horizon_1": {"de": "Für morgen (D+1)", "en": "For tomorrow (D+1)"},
    "horizon_2": {"de": "Für übermorgen (D+2)", "en": "For the day after (D+2)"},
    "col_days": {"de": "Tage fällig / prognostiziert", "en": "Days due / forecast"},
    "col_mae": {"de": "MAE", "en": "MAE"},
    "col_rmse": {"de": "RMSE", "en": "RMSE"},
    "col_skill": {"de": "Skill", "en": "Skill"},
    "col_pinball": {"de": "Pinball", "en": "Pinball"},
    "col_cov": {"de": "im 80-%-Band", "en": "inside 80 % band"},
    "reference": {"de": "Referenz", "en": "reference"},
    "units_note": {
        "de": "MAE, RMSE und Pinball in €/MWh je Viertelstunde (10 €/MWh = 1 ct/kWh). Skill = "
        "1 − MAE ÷ MAE der Referenz: über 0 heißt besser als die Faustregel. Beim 80-%-Band "
        "sollten etwa 80 % der Preise im Band liegen.",
        "en": "MAE, RMSE and pinball in EUR/MWh per quarter-hour (10 EUR/MWh = 1 ct/kWh). "
        "Skill = 1 − MAE ÷ MAE of the reference: above 0 means better than the rule of thumb. "
        "About 80 % of prices should fall inside the 80 % band.",
    },
    "segments_title": {"de": "Nach Situation", "en": "By situation"},
    "seg_all": {"de": "Alle Viertelstunden", "en": "All quarter-hours"},
    "seg_weekday": {"de": "Werktage", "en": "Weekdays"},
    "seg_weekend": {"de": "Wochenende", "en": "Weekend"},
    "seg_negative_price": {"de": "Negative Preise", "en": "Negative prices"},
    "seg_price_spike": {"de": "Preisspitzen", "en": "Price spikes"},
    "spike_def": {
        "de": "Preisspitze: tatsächlicher Preis ab {x} €/MWh (oberste 10 % im Zeitraum).",
        "en": "Price spike: actual price of {x} EUR/MWh or more (top 10 % in the window).",
    },
    "dm_title": {"de": "Ist der Unterschied echt?", "en": "Is the difference real?"},
    "dm_lede": {
        "de": "Diebold-Mariano-Test auf die täglichen Fehler, paarweise, zweiseitig. Ein kleiner "
        "p-Wert heißt: Der Unterschied ist kaum Zufall. Mit wenigen Tagen ist der Test schwach.",
        "en": "Diebold-Mariano test on daily errors, pairwise, two-sided. A small p-value means "
        "the difference is unlikely to be chance. With few days the test is weak.",
    },
    "dm_better": {"de": "{a} ist besser als {b}", "en": "{a} is better than {b}"},
    "dm_none": {"de": "kein belegter Unterschied", "en": "no demonstrated difference"},
    "dm_few": {"de": "zu wenige Tage ({n})", "en": "too few days ({n})"},
    "chart_mae_title": {"de": "Täglicher Fehler (MAE, D+1)", "en": "Daily error (MAE, D+1)"},
    "cheap_record_title": {
        "de": "Trefferbilanz der günstigsten 3 Stunden",
        "en": "Track record of the cheapest 3 hours",
    },
    "col_hit": {"de": "Treffer", "en": "Hit rate"},
    "col_p_stated": {"de": "angegeben", "en": "stated"},
    "col_near": {"de": "höchstens 0,5 ct teurer", "en": "within 0.5 ct"},
    "col_regret": {"de": "Mehrkosten ggü. bestem Fenster", "en": "Extra cost vs. best window"},
    "col_saving": {"de": "Ersparnis ggü. Tagesmittel", "en": "Saving vs. day mean"},
    "missed_title": {"de": "Verpasste Prognosen", "en": "Missed forecasts"},
    "missed_none": {"de": "Keine verpassten Prognosen.", "en": "No missed forecasts."},
    "backtest_title": {
        "de": "Backtest – nicht Teil der Bilanz",
        "en": "Backtest – not part of the track record",
    },
    "backtest_lede": {
        "de": "Derselbe Code, rückwirkend für {n} vergangene Tage ({from_} bis {to}), jeweils nur "
        "mit den Daten, die damals bekannt waren. Er ist nicht vorab festgelegt und deshalb "
        "optimistisch. Er zeigt, warum ein Modell live läuft, nicht dass es gut ist.",
        "en": "The same code, run for {n} past days ({from_} to {to}) with only the data known "
        "then. It is not pre-registered and therefore optimistic. It shows why a model runs "
        "live, not that it is good.",
    },
    # failures
    "fail_title": {"de": "Wo das Modell versagt", "en": "Where the model fails"},
    "fail_lede": {
        "de": "Die Tage mit dem größten Fehler, mit automatisch erzeugten Hinweisen aus den "
        "Preisen selbst. Das sind Anhaltspunkte, keine Ursachenanalyse: Wind, Sonne, Gas "
        "und Kraftwerksausfälle sieht das Modell nicht.",
        "en": "The days with the largest error, with hints generated automatically from the "
        "prices themselves. They are clues, not a root-cause analysis: the model does not "
        "see wind, sun, gas or plant outages.",
    },
    "fail_live": {"de": "Live: {model}", "en": "Live: {model}"},
    "fail_live_empty": {
        "de": "Noch keine bewerteten Live-Tage.",
        "en": "No scored live days yet.",
    },
    "fail_backtest": {"de": "Aus dem Backtest", "en": "From the backtest"},
    "fail_card_head": {
        "de": "{model}, {horizon}: MAE {mae} €/MWh",
        "en": "{model}, {horizon}: MAE {mae} EUR/MWh",
    },
    "h_short_1": {"de": "D+1", "en": "D+1"},
    "h_short_2": {"de": "D+2", "en": "D+2"},
    # explanations
    "ex_level": {
        "de": "Das Preisniveau sprang: Tagesmittel {now} €/MWh statt {before} am letzten bekannten "
        "Tag.",
        "en": "The price level jumped: day mean {now} EUR/MWh instead of {before} on the last "
        "known day.",
    },
    "ex_negative": {
        "de": "{n} Viertelstunden mit negativen Preisen, Minimum {min} €/MWh.",
        "en": "{n} quarter-hours with negative prices, minimum {min} EUR/MWh.",
    },
    "ex_spike": {
        "de": "Preisspitze bis {max} €/MWh um {time}, höher als an 95 % der Tage davor.",
        "en": "Price spike up to {max} EUR/MWh at {time}, higher than on 95 % of the days before.",
    },
    "ex_spread": {
        "de": "Ungewöhnlich große Spanne innerhalb des Tages: {spread} €/MWh (sonst etwa {usual}).",
        "en": "Unusually wide range within the day: {spread} EUR/MWh (usually about {usual}).",
    },
    "ex_holiday": {"de": "Feiertag: {name}.", "en": "Public holiday: {name}."},
    "ex_after_holiday": {
        "de": "Tag nach einem Feiertag ({name}).",
        "en": "Day after a public holiday ({name}).",
    },
    "ex_dst": {"de": "Tag der Zeitumstellung.", "en": "Daylight saving time switch."},
    "ex_bias_high": {
        "de": "Die Prognose lag im Mittel {x} €/MWh zu hoch.",
        "en": "The forecast was {x} EUR/MWh too high on average.",
    },
    "ex_bias_low": {
        "de": "Die Prognose lag im Mittel {x} €/MWh zu niedrig.",
        "en": "The forecast was {x} EUR/MWh too low on average.",
    },
    "ex_worst_time": {
        "de": "Größter Fehler um {time}: Prognose {f}, tatsächlich {a} €/MWh.",
        "en": "Largest error at {time}: forecast {f}, actual {a} EUR/MWh.",
    },
    "ex_none": {
        "de": "Kein auffälliges Muster in den Preisen selbst. Wahrscheinlich wichen Wind oder "
        "Sonne von den Vortagen ab.",
        "en": "No striking pattern in the prices themselves. Most likely wind or sun differed "
        "from the previous days.",
    },
    # cheapest
    "cheap_title": {"de": "Günstige Stunden", "en": "Cheapest hours"},
    "cheap_lede": {
        "de": "Für dynamische Stromtarife: Wann lohnt es sich, Waschmaschine, Wärmepumpe oder "
        "E-Auto laufen zu lassen? Die Angaben sind Prognosen mit Unsicherheit, keine "
        "Garantie.",
        "en": "For dynamic electricity tariffs: when is it worth running the washing machine, "
        "heat pump or electric car? These are forecasts with uncertainty, not a guarantee.",
    },
    "cheap_when": {"de": "{day}, {start}–{end} Uhr", "en": "{day}, {start}–{end}"},
    "cheap_expected": {
        "de": "Erwarteter Börsenpreis im Fenster {win} ct/kWh, im Tagesmittel {day} ct/kWh.",
        "en": "Expected exchange price in the window {win} ct/kWh, day mean {day} ct/kWh.",
    },
    "cheap_scen": {
        "de": "In {pc} von 100 Szenarien ist dieses Fenster das günstigste. In {pn} von 100 ist "
        "es höchstens 0,5 ct/kWh teurer als das günstigste.",
        "en": "In {pc} of 100 scenarios this window is the cheapest. In {pn} of 100 it costs at "
        "most 0.5 ct/kWh more than the cheapest.",
    },
    "cheap_alts": {"de": "Andere Kandidaten", "en": "Other candidates"},
    "cheap_note": {
        "de": "Das ist der Börsenpreis ohne Steuern, Umlagen und Netzentgelte. Die kommen bei "
        "dynamischen Tarifen je kWh hinzu, die Reihenfolge der Stunden bleibt aber gleich. Die "
        "Szenarien entstehen aus den echten Prognosefehlern der letzten 60 bis 90 Tage. Die "
        "Wahrscheinlichkeiten werden vor der Auktion gespeichert und später geprüft; die "
        "Trefferbilanz steht unten.",
        "en": "This is the exchange price without taxes, levies and grid fees. With dynamic "
        "tariffs these are added per kWh, but the order of the hours stays the same. Scenarios "
        "come from the real forecast errors of the last 60 to 90 days. Probabilities are saved "
        "before the auction and checked later; the track record is below.",
    },
    "cheap_empty": {
        "de": "Sobald die erste Prognose ausgegeben ist, steht hier das günstigste Fenster.",
        "en": "The cheapest window appears here once the first forecast is issued.",
    },
    # data
    "data_title": {"de": "Daten", "en": "Data"},
    "data_lede": {
        "de": "Alle Prognosen, Preise und Bewertungen zum Herunterladen, täglich aktualisiert. "
        "Daten unter CC BY 4.0, Preise von der Bundesnetzagentur | SMARD.de.",
        "en": "All forecasts, prices and scores for download, updated daily. Data under CC BY "
        "4.0, prices from Bundesnetzagentur | SMARD.de.",
    },
    "file": {"de": "Datei", "en": "File"},
    "content": {"de": "Inhalt", "en": "Content"},
    "rows": {"de": "Zeilen", "en": "Rows"},
    "size": {"de": "Größe", "en": "Size"},
    "dictionary": {"de": "Spaltenbeschreibung", "en": "Column dictionary"},
    "powerbi_title": {"de": "In Power BI oder Excel laden", "en": "Load into Power BI or Excel"},
    "powerbi_body": {
        "de": "CSV-Dateien sind UTF-8, durch Kommas getrennt, mit Punkt als Dezimaltrennzeichen. "
        "In Power BI: Daten abrufen → Text/CSV → Dateiursprung „65001: Unicode (UTF-8)“, "
        "Trennzeichen Komma. Danach unter „Typ ändern → Mit Gebietsschema…“ Englisch (USA) "
        "wählen, damit Dezimalzahlen richtig gelesen werden. Zeiten gibt es in UTC und in "
        "deutscher Ortszeit; für Tagesauswertungen die Spalte mit dem lokalen Datum nehmen.",
        "en": "CSV files are UTF-8, comma-separated, with a dot as decimal separator. In Power "
        "BI: Get data → Text/CSV → file origin '65001: Unicode (UTF-8)', delimiter comma. With a "
        "non-English locale use 'Change type → Using locale…' English (United States). Times "
        "are given in UTC and in German local time; use the local date column for daily views.",
    },
    "latest_title": {"de": "Für Programme", "en": "For programs"},
    "latest_body": {
        "de": "latest.json enthält die jüngste Prognose aller Modelle samt günstigster Fenster und "
        "Bilanz-Kennzahlen. Die unveränderlichen Originaldateien liegen im Repository.",
        "en": "latest.json holds the most recent forecast of all models, the cheapest windows and "
        "the headline scores. The immutable original files are in the repository.",
    },
    "method_title": {"de": "Methodik", "en": "Methodology"},
    "nav_imprint": {"de": "Impressum und Datenschutz", "en": "Legal notice and privacy"},
    "imprint_title": {"de": "Impressum", "en": "Legal notice"},
    "imprint_by": {"de": "Angaben gemäß § 5 DDG", "en": "Information according to § 5 DDG"},
    "imprint_contact": {"de": "Kontakt", "en": "Contact"},
    "imprint_form": {"de": "Kontaktformular", "en": "Contact form"},
    "imprint_resp": {
        "de": "Verantwortlich für den Inhalt nach § 18 Abs. 2 MStV",
        "en": "Responsible for content according to § 18 (2) MStV",
    },
    "imprint_same": {
        "de": "Till Oscar Jacob, Anschrift wie oben",
        "en": "Till Oscar Jacob, address as above",
    },
    "imprint_links_title": {"de": "Haftung für Links", "en": "Liability for links"},
    "imprint_links": {
        "de": "Diese Website enthält Links zu externen Websites Dritter, auf deren Inhalte ich "
        "keinen Einfluss habe. Für diese fremden Inhalte ist stets der jeweilige Anbieter oder "
        "Betreiber der Seiten verantwortlich. Bei Bekanntwerden von Rechtsverletzungen werde ich "
        "derartige Links umgehend entfernen.",
        "en": "This website links to external third-party websites whose content I cannot "
        "influence. The respective provider or operator is always responsible for that content. "
        "If I become "
        "aware of legal violations, I will remove such links immediately.",
    },
    "privacy_title": {"de": "Datenschutz", "en": "Privacy"},
    "privacy_body": {
        "de": "Diese Seite setzt keine Cookies, verwendet kein Tracking und lädt keine Inhalte von "
        "Dritten; Schriften und Skripte liegen auf demselben Server. Gehostet wird sie von GitHub "
        "Pages (GitHub Inc.). Beim Aufruf verarbeitet GitHub technisch notwendige Daten wie die "
        "IP-Adresse, um die Seite auszuliefern und abzusichern; Einzelheiten stehen in der "
        "Datenschutzerklärung von GitHub. Ich selbst erhalte keine Besucherdaten.",
        "en": "This site sets no cookies, uses no tracking and loads no third-party content; fonts "
        "and scripts are served from the same host. It is hosted on GitHub Pages (GitHub Inc.). "
        "When you visit, GitHub processes technically necessary data such as your IP address to "
        "deliver and secure the site; see GitHub's privacy statement. I receive no visitor data.",
    },
    "dict_table": {"de": "Tabelle", "en": "Table"},
    "dict_column": {"de": "Spalte", "en": "Column"},
    "dict_type": {"de": "Typ", "en": "Type"},
    "dict_unit": {"de": "Einheit", "en": "Unit"},
    "dict_desc": {"de": "Beschreibung", "en": "Description"},
    # footer
    "footer_disclaimer": {
        "de": "Keine Anlageberatung, keine Handelssignale, kein Versprechen von Ersparnissen.",
        "en": "Not investment advice, not a trading signal, no promise of savings.",
    },
    "footer_source": {
        "de": "Preise: Bundesnetzagentur | SMARD.de (CC BY 4.0). Prognosen und Bewertungen: CC BY "
        "4.0. Code: MIT.",
        "en": "Prices: Bundesnetzagentur | SMARD.de (CC BY 4.0). Forecasts and scores: CC BY 4.0. "
        "Code: MIT.",
    },
    "footer_privacy": {
        "de": "Diese Seite setzt keine Cookies, lädt keine fremden Inhalte und zählt keine "
        "Besucher. Gehostet von GitHub Pages.",
        "en": "This site sets no cookies, loads no third-party content and does not count "
        "visitors. Hosted on GitHub Pages.",
    },
    "footer_built": {"de": "Stand: {t}", "en": "Updated: {t}"},
    "values_table": {"de": "Werte als Tabelle", "en": "Values as a table"},
    "time": {"de": "Zeit", "en": "Time"},
}


def t(key: str, lang: str, **kw: Any) -> str:
    s = T[key][lang]
    return s.format(**kw) if kw else s
