# Hinweise für Claude

## Pflicht bei jeder Änderung

- **`CHANGELOG.md` führen.** Jede Änderung am Repo bekommt im selben Commit einen Eintrag oben in `CHANGELOG.md`: was geändert wurde und warum, dazu „Geprüft“ (was wirklich gelaufen ist, mit Python-Version und Testzahl) und „Offen“ (was nicht geprüft wurde). Sprache: Deutsch, einfache Sätze.
- Wenn sich Testzahlen oder der Prüfstand ändern, auch die Tabelle „Stand“ in `README.md` anpassen.

## Arbeitsweise

- Eigener Branch, kleine Commits mit Tests, am Ende ein Pull Request, den der Nutzer mergt.
- Vor jedem Commit: `ruff check .`, `ruff format --check .`, `pytest -q --cov`.
- Optionale Bibliotheken: `pip install -e ".[dev,shapiq]"`, torch als CPU-Wheel, probly mit `pip install "probly @ git+https://github.com/pwhofman/probly@2729f4f"`.
- Der Nutzer arbeitet unter Windows mit PowerShell und ist Einsteiger in Git: Schritte einfach erklären, Befehle zum Kopieren geben.
