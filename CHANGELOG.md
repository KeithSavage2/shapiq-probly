# Änderungsprotokoll

Hier steht jede Änderung am Repo: **was** geändert wurde und **warum**. Neue Einträge kommen oben dazu. Jede Änderung, auch eine kleine, bekommt im selben Commit einen Eintrag.

Aufbau eines Eintrags: Datum, Branch, dann pro Änderung „Was“ und „Warum“, am Ende „Geprüft“ (was wirklich gelaufen ist) und „Offen“ (was nicht).

---

## 13.11.2026 – Tests und CI ausgebaut

Branch `claude/tests-ci-changelog`

### Lücken in den Tests geschlossen

- **Was:** Neue Datei `tests/test_edge_cases.py` mit 16 Tests für Zweige und Fehlermeldungen, die bisher kein Test erreicht hat: Auffüllen fehlender Klassen, Regressoren mit und ohne eigene Varianz, Regressionsform `(M, n, 1)`, eigene Maße als Funktion, Zahlenformat, die drei Lesarten in `summary()`, der Hinweis im Unsicherheitsprofil bei mehr als 6 Merkmalen und 15 Fehlermeldungen.
- **Warum:** Die Abdeckung lag bei 90 %, und gerade die Fehlermeldungen waren ungetestet. Eine Fehlermeldung, die nie ausgelöst wurde, kann selbst kaputt sein.
- **Was dabei auffiel:** Der alte Test „missing classes“ hat das Auffüllen nie erreicht. sklearns `BaggingClassifier` gibt Entscheidungsbäumen alle Klassen mit (mit Gewicht 0), also sieht dort jeder Member jede Klasse. Der neue Test trainiert einen Member von Hand auf nur zwei von drei Klassen.

### Ground Truth mit mehreren Seeds und Modelltypen

- **Was:** Neuer Test in `tests/test_synthetic.py`: Seeds 1 bis 4, jeweils Random Forest, Extra Trees und Bagging (12 Läufe). Pro Seed neue Daten, neues Modell, neue Imputation.
- **Warum:** Die Befunde standen bisher auf einem Seed und einem Modelltyp. Das reicht nicht, um sie in der Präsentation zu verteidigen.
- **Ergebnis:** In allen 12 Läufen stabil: `pair_a x pair_b` ist die stärkste epistemische Interaktion am Lückenpunkt (k-SII 0,17 bis 0,36), das Rauschmerkmal ist der stärkste aleatorische Treiber am Rauschpunkt, das Dummy-Merkmal bleibt unter 0,05, die Additivität hält.
- **Nicht stabil, deshalb nicht im neuen Test:** „Der sichere Punkt hat fast keine epistemische Unsicherheit“ (Random Forest, Seed 4: EU 0,16 statt etwa 0,01) und die Reihenfolge der aleatorischen Unsicherheit zwischen Lückenpunkt und sicherem Punkt. Die alten Tests mit Seed 0 prüfen das weiter; es gilt also nur für diesen Seed.

### CI ausgebaut

- **Was:** `.github/workflows/ci.yml` hat jetzt sechs Test-Jobs (core, shapiq, torch, probly, core auf Windows, core auf Python 3.13) und einen Job, der die drei Beispiele ausführt. Jeder Test-Job misst die Abdeckung, zeigt die nicht erreichten Zeilen und schlägt unter 85 % fehl. Dazu `ruff format --check`. `pytest-cov` ist neu in der Gruppe `dev`.
- **Warum:** Der Nutzer arbeitet unter Windows, getestet wurde aber nur auf Ubuntu. Die Beispiele liefen in keiner Prüfung mit und konnten unbemerkt brechen. Ohne Abdeckungsmessung sieht man nicht, was ungetestet ist.
- **Warum 85 %:** Jeder Job installiert nur eine der optionalen Bibliotheken; die Brücken zu den anderen zählen dort als ungetestet. Der Kern-Job kommt lokal auf 92 %.

### Dokumentation

- **Was:** Dieses Protokoll und `CLAUDE.md` angelegt, README-Tabelle „Stand“ aktualisiert.
- **Warum:** Der Nutzer will jede Änderung mit Begründung im Repo nachlesen können. `CLAUDE.md` wird von Claude zu Beginn jeder Sitzung gelesen und enthält die Regel, dieses Protokoll zu führen.

### Geprüft

- Python 3.13, Linux, mit shapiq 1.8.0 und probly `2729f4f`: 191 Tests grün, Abdeckung 94 %, `ruff check` und `ruff format --check` sauber.
- Ohne shapiq und probly: 181 grün, 7 übersprungen, Abdeckung 92 %.
- Die drei Beispiele laufen durch (3 s, 7 s, 13 s).

### Offen

- Die neue CI ist noch nie auf GitHub gelaufen. Der Windows-Job, der Python-3.12-Lauf und der torch-Job sind also unbestätigt.
- Weiter ohne Test: `torch_bridge.py` außerhalb des torch-Jobs, die beiden Basisklassen-Methoden, die nur `NotImplementedError` werfen, und `measures.py` Zeile 124–125 (nach heutigem Stand nicht erreichbar).

---

## 13.11.2026 – `from_probly` repariert und gegen echtes probly getestet

Branch `claude/fix-probly-bridge`, Commit `a93e88a`

- **Was:** `_default_to_probs` in `probly_bridge.py` liest jetzt `array.probabilities` und schiebt die Achse `sample_axis` nach vorn.
- **Warum:** probly gibt `(n, M, C)` zurück (Batch zuerst, Member an zweiter Stelle). Der alte Code nahm `.samples` (schon Member-zuerst) und tauschte noch einmal; `UncertaintyGame` brach mit `ValueError` ab. Die Brücke war damit unbenutzbar.
- **Was:** Ein Modell ohne Member (z. B. `representer(RandomForestClassifier)`) wird mit einer Fehlermeldung abgelehnt, die auf probly's `ensemble` verweist.
- **Warum:** Vorher kam eine unverständliche Meldung über Achsen.
- **Was:** `tests/test_probly_bridge.py` mit 9 Tests: 4 mit einem Stellvertreter-Objekt (laufen immer), 5 gegen echtes probly (übersprungen, wenn probly fehlt). Geprüft wird: Member-Vorhersagen, volle Koalition und Entropie-Zerlegung stimmen mit `probly.quantification.quantify` überein (Abweichung unter 1e-9).
- **Warum:** Das Modul hatte 0 % Abdeckung.
- **Was:** CI-Job, der probly von GitHub installiert (fest auf Commit `2729f4f`).
- **Warum:** probly ändert sich täglich; ein fester Commit hält die Tests reproduzierbar.
- **Was:** Veraltete Texte korrigiert (README-Tabelle „Stand“, Docstrings von `torch_bridge.py`, `shapiq_bridge.py`, `tests/test_torch_bridge.py`).
- **Warum:** Dort stand noch, torch und shapiq seien nie ausgeführt worden; beide laufen seit v0.2 in der CI.
- **Geprüft:** Gemergt als PR #2 (Merge-Commit `8998275`). Auf GitHub alle vier CI-Jobs grün (Python 3.12, Ubuntu), darunter zum ersten Mal der Job mit probly.
- **Offen:** Nur Klassifikation. Für sklearn-Regressoren liefert probly keine Varianz-Zerlegung.
