<!--
ENTWURF. Erst veröffentlichen, wenn mindestens vier Wochen Live-Bilanz vorliegen.
Alle Werte in [[…]] aus `uv run prognosebuch headline` übernehmen (nur Live-Zahlen, keine
Backtest-Zahlen). Wenn das beste Modell live NICHT besser ist als die Faustregel, den zweiten Absatz
entsprechend umschreiben: Das ist dann das Ergebnis.
-->

Kann man den Strompreis von morgen vorhersagen? Und woher weiß man, ob die Vorhersage etwas taugt?

Seit dem 30. September lege ich jeden Morgen vor 12 Uhr eine Prognose der Day-Ahead-Preise für Deutschland ab, für jede Viertelstunde der nächsten zwei Tage, mit Unsicherheitsband. Jede Prognose wird als unveränderliche Datei öffentlich gespeichert, bevor die Börse die Preise veröffentlicht, und am Nachmittag automatisch bewertet. Verpasste Tage zählen mit.

Nach [[TAGE]] Tagen:
• Mein bestes Modell ([[MODELL]], z. B. Gradient Boosting mit Wetterprognosen) lag im Mittel [[MAE_BEST_D1]] €/MWh daneben, die einfache Faustregel „wie gestern bzw. wie letzte Woche" [[MAE_REF_D1]] €/MWh.
• Das sind [[SKILL_PROZENT]] % weniger Fehler. Der Diebold-Mariano-Test sagt: [[DM_SATZ]].
• Verpasst: [[VERPASST]] Prognosen.
• Am schlechtesten sind alle Modelle bei negativen Preisen und Preisspitzen. Warum, steht auf der Seite „Wo das Modell versagt".

Was ich dabei gelernt habe: Ein Backtest ist leicht schön zu rechnen. Ehrlich wird eine Prognose erst, wenn man sie vorher festlegt und danach nichts mehr ändern kann. Dafür sorgen hier eine CI-Regel, Prüfsummen und automatische Bewertung.

Alles ist offen und kostet 0 € Betrieb: Python, scikit-learn, GitHub Actions, Daten von SMARD (Bundesnetzagentur) und Open-Meteo. Für Nutzer dynamischer Tarife zeigt die Seite außerdem die günstigsten drei Stunden, mit einer Wahrscheinlichkeit, die ebenfalls vorab festgehalten und geprüft wird.

Live-Bilanz: https://tojacob03.github.io/Prognosebuch/
Code und Daten: https://github.com/tojacob03/Prognosebuch

Keine Anlageberatung, keine Handelssignale. Feedback, gerade von Leuten aus Energiehandel und Data Science, ist sehr willkommen.

#Datenanalyse #Energiewende #Python #Prognose #OpenData
