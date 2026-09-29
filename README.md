# Agentic Oracle Resolver for Prediction Markets

Материалы исследования для отбора в «Институт Вега»: обзор литературы, аудит референсной работы, гипотезы и первые эксперименты с данными.

- [Досье](research_dossier.md) — исследовательский вопрос, литература и план.
- [Notebook](first_steps.ipynb) — выполненные расчёты и график.
- [Аудит](audit/audit_findings.txt) — проверка метрик и ограничений сравнения.
- `data/kalshi/` — 100 рынков с раздельными входами и ответами.
- `src/`, `results/` — скрипты, метрики и экономические расчёты.
- `sources/`, `vendor/reference/` — источники и снимок кода автора с лицензией; происхождение и SHA-256 в `vendor/reference_source.json`.

**Главная находка:** точность B до обсуждения — 76,62%, после — 76,11%. Разрыв с независимым ансамблем нельзя целиком считать эффектом debate.

## Запуск

Python 3.10+, без платных API:

```bash
python3 audit/audit_public_artifacts.py
python3 audit/audit_cache_coverage.py
python3 src/risk_and_cost.py
```

Для пересоздания notebook: `pip install -r requirements-notebook.txt`, затем `python3 src/build_notebook.py`.

Новые LLM-запуски не выполнялись. Выгрузка Kalshi — техническая проба, не репрезентативный benchmark; денежные параметры иллюстративны. Evidence cache референса сохранён частично; полная привязка к историческим ответам не подтверждена.
