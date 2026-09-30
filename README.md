# Agentic Oracle Resolver for Prediction Markets

Аудит референсной работы, исследовательские гипотезы и подготовленные эксперименты для проекта «Института Вега».

- [Досье](research_dossier.md) — выводы и план исследования; [литература](docs/literature.md).
- [Notebook](first_steps.ipynb) — выполненные расчёты и графики.
- [Направление ошибок](audit/error_direction.json), [основной аудит](audit/audit_results.json), [экономика и размеры выборки](results/risk_and_cost.json).
- [Абляция](experiments/date_ablation_protocol.json) — явная дата и содержимое evidence; генератор 600/1200 запросов.
- `data/kalshi/` — техническая выборка из 100 рынков, входы отдельно от labels.

**Результат:** разрыв A–B возникает до debate и состоит преимущественно из ложных NO. Обнаружены различия промптов и форматирования evidence; причинный эффект ещё не установлен. Новые LLM-вызовы не выполнялись.

## Воспроизведение

Python 3.11, make, curl, venv; платные API и ключи не нужны.

```bash
make fetch       # скачать закреплённые источники и проверить SHA-256
make reproduce   # офлайн: аудит, риск, экономика, подготовка запросов
```

Исходники автора и KalshiBench скачиваются локально, не входят в текущее дерево Git. Происхождение: [sources_manifest.json](sources_manifest.json), [reference_source.json](vendor/reference_source.json). Pyarrow используется только на этапе fetch; расчёты — stdlib. CI проверяет воспроизведение без изменения отслеживаемых результатов.

Notebook: `python3 -m pip install -r requirements-notebook.txt`, затем `python3 src/build_notebook.py`. Параметры экономики иллюстративны; выборка Kalshi не является репрезентативным benchmark.
