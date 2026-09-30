# Methodik

Was wird prognostiziert, wann, mit welchen Informationen, wie wird bewertet, und was sagen die
Zahlen aus und was nicht. Die englische Fassung in `METHODOLOGY.md` im Repository ist die
Referenz; diese Seite gibt sie auf Deutsch wieder.

**Keine Anlageberatung, keine Handelssignale, kein Versprechen von Ersparnissen.**

## 1. Zielgröße

Der Day-Ahead-Preis der Gebotszone Deutschland-Luxemburg (DE-LU) aus der Auktion der EPEX SPOT,
wie ihn die Bundesnetzagentur auf SMARD.de veröffentlicht. Seit dem 1. Oktober 2025 wird in
Viertelstunden gehandelt; alle Prognosen und Bewertungen sind je Viertelstunde. Stundenwerte
sind das Mittel der vier Viertelstunden.

Ein *Liefertag* ist ein Kalendertag in deutscher Zeit: 96 Viertelstunden, am letzten
Märzsonntag 92, am letzten Oktobersonntag 100.

## 2. Zeitpunkte und Datenschnitt

| Deutsche Zeit | Ereignis |
|---|---|
| D−1, ca. 12:45 | Preise für Tag D werden veröffentlicht |
| **D, 08:45–12:00** | **Prognosefenster. Prognosen für D+1 und D+2 (Ziel etwa 09:00)** |
| D, 12:00 | Gebotsschluss der Auktion für D+1, harte Frist |
| D, ca. 12:45 | Preise für D+1 erscheinen; die D+1-Prognose kann bewertet werden |
| D+1, ca. 12:45 | Preise für D+2 erscheinen; die D+2-Prognose kann bewertet werden |

Der **Prognosezeitpunkt** ist der Moment, in dem die Prognosedatei entstand (`issued_at_utc` im
Manifest). Der **Datenschnitt** ist der letzte Zeitpunkt, dessen Daten ein Modell nutzen darf.
Bei Preisen ist das das Ende von Tag D: Alle Viertelstunden von D sind bekannt, weil sie am
Vortag veröffentlicht wurden, von D+1 ist nichts bekannt.

Diese Regeln sind im Code erzwungen und getestet:

- Der Job prognostiziert nur im Fenster; nach 12:00 verweigert er, und der Tag zählt als verpasst.
- Zeigt SMARD schon einen Preis für D+1, verweigert der Job ebenfalls.
- Modelle erhalten nur ein Informationspaket, das alle Preise nach dem Datenschnitt hart
  abschneidet. Ein Test setzt alle späteren Preise auf einen absurden Wert und prüft, dass jede
  Prognose bitgenau gleich bleibt.

GitHubs Zeitplaner hat sich für dieses Repository als unzuverlässig erwiesen (Läufe bis zu sechs
Stunden zu spät oder gar nicht; der erste Ausgabetag ging dadurch verloren, siehe
`INCIDENTS.md`). Deshalb wartet ein Workflow `clock` auf die Termine in
`src/prognosebuch/clock.py` und startet den Prognose-Job um 08:55, 09:30 und 10:30 deutscher
Zeit; vor GitHubs Sechs-Stunden-Grenze übergibt er an einen neuen Lauf von sich selbst. Die
Cron-Einträge bleiben als Reserve. Der Job prüft die deutsche Uhrzeit selbst; nur der erste
erfolgreiche Lauf des Tages schreibt, spätere Läufe ergänzen nur fehlende Modelle.

## 3. Das Buch (Live-Bilanz)

- Jede Prognose wird genau einmal als Datei geschrieben:
  `forecasts/JJJJ/MM/<Ausgabetag>/<Modell>.v<Version>.parquet` plus ein JSON-Manifest mit
  Ausgabezeit, Datenschnitt je Quelle, Code-Commit, Link zum Workflow-Lauf und SHA-256-Prüfsumme.
- Der Bot committet nur neue Dateien. Die Prüfung `prognosebuch audit` läuft bei jedem Push,
  täglich und nach jedem Job; sie schlägt fehl, wenn eine Prognosedatei jemals geändert oder
  gelöscht wurde oder Prüfsumme oder Ausgabezeit nicht passen.
- Der Bewertungsjob bewertet jeden Liefertag, sobald seine Preise vollständig sind und die
  letzte Frist vorbei ist. Jede fällige, aber fehlende Prognose wird als `missed` eingetragen.
  Verpasste Tage zählen mit und bleiben sichtbar.
- Ein Modell hat Name und Version. Jede Änderung der Logik ergibt eine neue Version, die parallel
  läuft; die Bilanz aller Versionen bleibt sichtbar.

Was Außenstehende prüfen können: Prüfsummen, die Git-Historie von `forecasts/` (nur
Hinzufügungen), den im Manifest verlinkten Workflow-Lauf samt Log und Zeit und, sobald
eingerichtet, wöchentliche Releases mit DOI. Git-Zeitstempel allein beweisen nichts; die
Push-Zeit auf GitHub und die Logs schon.

## 4. Modelle

| Modell | Idee |
|---|---|
| `naive_last_day.v1` | Tag D auf den Zieltag kopieren |
| `naive_weekly.v1` | Gleicher Wochentag eine Woche vor dem Zieltag |
| `naive_similar_day.v1` | **Referenz.** Standard aus der Literatur (Lago et al. 2021): Mo/Sa/So wie vor einer Woche, Di–Fr wie der Vortag (bei D+2 der letzte bekannte Tag) |
| `lear.v1` | Lasso-geschätzte Autoregression (siehe unten) |
| `gbm.v1` | Gradient Boosting mit Quantilverlust auf Wetterprognosen, verzögerten Preisen und Kalender (siehe unten) |

Kopiert wird nach Uhrzeit, deshalb funktionieren Tage mit Zeitumstellung.

**LEAR** (eigene Umsetzung nach Lago, Marcjasz, De Schutter und Weron, 2021):

- Je Lieferstunde und Horizont ein lineares Modell auf Stundenmittelwerten.
- Eingaben: die 24 Stundenpreise der Tage `t−h`, `t−h−1`, `t−h−2` und `t−7`, Wochentage,
  bundesweite Feiertage.
- Transformation `asinh((x − Median) / MAD)` je Spalte.
- L1-Strafterm je Stunde nach dem Akaike-Informationskriterium auf dem LARS-Pfad.
- Mittel aus drei Kalibrierungsfenstern (182, 364, 728 Tage), täglich neu geschätzt.
- Viertelstunden: Stundenprognose plus das mittlere Profil innerhalb der Stunde der letzten 28 Tage.

**Gradient Boosting** (`gbm.v1`): histogrammbasiertes Gradient Boosting aus scikit-learn
(dieselbe Verfahrensfamilie wie LightGBM, gewählt, weil es ohne native OpenMP-Bibliothek
auskommt) mit Quantilverlust, je ein Modell für P10, P50 und P90 und je Horizont, über alle
24 Stunden gemeinsam.

- Wetter: Open-Meteo Previous Runs des DWD-Modells ICON an 12 Punkten (Wind in 100 m Höhe in
  Wind-Regionen an Land und auf See, Sonneneinstrahlung, Temperatur), zusammengefasst zu einem
  Windleistungs-Näherungswert, mittlerer Windgeschwindigkeit, Einstrahlung und Temperatur.
  D+1 nutzt Werte, die 48 h vor dem Zeitpunkt vorhergesagt wurden, D+2 72 h. Ein Wert gilt 6 h
  nach seinem Modelllauf als verfügbar; das Informationspaket weist alles zurück, was am
  Ausgabetag um 08:45 noch nicht verfügbar war.
- Dazu: Tagesmittel des Zieltags und des letzten bekannten Tags und ihre Differenz; verzögerte
  Preise (letzter bekannter Tag, ein Tag davor, eine Woche vor dem Zieltag; Mittel, Minimum,
  Maximum des letzten bekannten Tags; Sieben-Tage-Mittel); Stunde, Wochentag, Wochenende,
  Feiertag, Jahreszeit.
- Zielgröße ist der Preis minus Sieben-Tage-Mittel: Die Bäume lernen Form und Wettereffekt,
  das Niveau kommt aus den jüngsten Preisen.
- Wöchentlich neu geschätzt auf allen Tagen bis zum letzten Sonntag; Training ab März 2024
  (das Archiv der Previous Runs beginnt im Februar 2024).
- P10 und P90 kommen direkt aus den Quantilmodellen; überkreuzen sie sich, werden sie sortiert.

**Unsicherheitsband (Faustregeln und LEAR).** P10 und P90 sind die Punktprognose plus das 10- bzw.
90-%-Quantil der eigenen Fehler des Modells an den letzten 60 (LEAR) bzw. 90 (Faustregeln)
Zieltagen, je Stunde und Horizont. Jede frühere Prognose wird dabei genau so neu berechnet, wie
sie an ihrem Ausgabetag entstanden wäre. Das Band nimmt an, dass die jüngste Fehlerverteilung
auch morgen gilt; es weiß nicht, ob morgen ein windiger Tag ist.

## 5. Kennzahlen

Je Modellversion, Horizont (D+1, D+2) und Zeitraum (7, 30, 90 Tage, gesamt):

| Kennzahl | Bedeutung |
|---|---|
| MAE | mittlerer absoluter Fehler in €/MWh |
| RMSE | Wurzel des mittleren quadratischen Fehlers; bestraft große Fehler stärker |
| Skill | 1 − MAE(Modell) ÷ MAE(Referenz) auf denselben Viertelstunden; über 0 = besser als die Faustregel |
| Pinball | mittlerer Quantilsverlust über P10, P50, P90; belohnt schmale *und* treffende Bänder |
| im 80-%-Band | Anteil der Preise in [P10, P90]; sollte etwa 0,80 sein |
| Tage fällig / prognostiziert | wie viele Tage fällig waren und wie viele eine Prognose hatten |

Getrennt ausgewiesen: alle Viertelstunden, Werktage, Wochenende, negative Preise, Preisspitzen
(tatsächlicher Preis in den obersten 10 % des Zeitraums).

**Diebold-Mariano-Test** (mit Korrektur nach Harvey, Leybourne und Newbold für kleine
Stichproben) auf die Reihe der täglichen MAE zweier Modelle, gepaart nach Liefertag, zweiseitig.
Ab 10 gemeinsamen Tagen. Mit wenigen Tagen ist „kein belegter Unterschied" das erwartbare Ergebnis.

## 6. Günstigste 3 Stunden

Für dynamische Tarife wird aus der Prognose je Tag eine Empfehlung: das 3-Stunden-Fenster
(12 aufeinanderfolgende Viertelstunden innerhalb des Liefertags) mit dem niedrigsten mittleren
Median. Die Wahrscheinlichkeiten kommen aus Szenarien: Median plus je eine der echten
Fehlerkurven des Modells der letzten 60 bis 90 Tage, nach Uhrzeit ausgerichtet. So bleibt die
Form echter Fehler innerhalb eines Tages erhalten.

- Wahrscheinlichkeit „günstigstes": Anteil der Szenarien, in denen das empfohlene Fenster das
  günstigste ist.
- Wahrscheinlichkeit „höchstens 0,5 ct teurer": Anteil der Szenarien, in denen es höchstens
  5 €/MWh über dem günstigsten liegt.

Beides wird vor der Auktion im Manifest gespeichert und danach bewertet: Trefferquote gegen die
angegebene Wahrscheinlichkeit (Kalibrierung), Brier-Score, Mehrkosten gegenüber dem besten
Fenster und Ersparnis gegenüber dem Tagesmittel. Es geht um Börsenpreise; Steuern, Umlagen und
Netzentgelte kommen je kWh hinzu und ändern die Reihenfolge der Stunden nicht.

## 7. Backtest und Live-Bilanz

Der Backtest rechnet denselben Code für jeden vergangenen Tag nur mit den damals bekannten Daten
nach; ein Test prüft, dass er die Live-Prognose exakt reproduziert. Er ist trotzdem **kein** Test
der Zukunft: Das Modell wurde von jemandem entworfen, der die Vergangenheit kannte. Backtest-Zahlen
stehen getrennt und sind gekennzeichnet. Als Nachweis zählt nur die Live-Bilanz.

## 8. Grenzen

- **Keine Gas- und CO₂-Preise.** Sie bestimmen das Preisniveau, aber es gibt keine offen
  lizenzierte Quelle. Niveauverschiebungen lernt das Modell nur über verzögerte Preise.
- **Noch keine Last- und Erneuerbaren-Prognosen der Netzbetreiber.** SMARD speichert keine alten
  Stände; unbekannt ist, was an vergangenen Tagen um 09:00 vorlag. Ein stündlicher Probe misst
  das. Solche Eingaben kommen erst hinzu, wenn sie nachweislich vor dem Prognosezeitpunkt vorliegen.
- Wetterprognosen kommen aus der Previous-Runs-API, mit so viel Vorlauf, dass die Daten zum
  Prognosezeitpunkt existierten (D+1: 48 h, D+2: 72 h). Die Historical-Forecast-API liegt nah
  an gemessenem Wetter und würde ein Datenleck erzeugen. Der Preis dieser Vorsicht: Live gäbe
  es einen frischeren Wetterlauf; das Modell nutzt ihn absichtlich nicht, damit Training und
  Live-Betrieb denselben Vorlauf haben.
- Die Bänder sind unbedingt (siehe oben), und Fehler innerhalb eines Tages hängen stark
  zusammen: Ein schlechter Tag ist meist über viele Stunden schlecht.
- Seltene Ereignisse (extreme Spitzen, Ausfälle, Störungen der Marktkopplung) lassen sich aus
  diesen Eingaben nicht vorhersagen.
