PYTHON ?= python3
FETCH_PYTHON = .fetch-venv/bin/python

.PHONY: fetch verify reproduce

# Network stage. pyarrow is used only to convert the pinned source Parquet.
fetch: .fetch-venv/.installed
	$(FETCH_PYTHON) src/fetch_sources.py

.fetch-venv/.installed: requirements-fetch.txt
	$(PYTHON) -m venv .fetch-venv
	$(FETCH_PYTHON) -m pip install --disable-pip-version-check -r requirements-fetch.txt
	touch $@

# Offline stage: stdlib Python 3.11, no models, credentials, or upstream code.
verify:
	$(PYTHON) src/fetch_sources.py --verify-only

reproduce: verify
	@mkdir -p .logs
	@date -u '+%Y-%m-%dT%H:%M:%SZ' > .logs/reproduce_started_utc.txt
	@$(PYTHON) audit/audit_public_artifacts.py > .logs/audit_public_artifacts.log 2>&1 || { cat .logs/audit_public_artifacts.log; exit 1; }
	@$(PYTHON) audit/audit_cache_coverage.py > .logs/audit_cache_coverage.log 2>&1 || { cat .logs/audit_cache_coverage.log; exit 1; }
	@$(PYTHON) audit/error_direction.py > .logs/error_direction.log 2>&1 || { cat .logs/error_direction.log; exit 1; }
	@$(PYTHON) src/risk_and_cost.py > .logs/risk_and_cost.log 2>&1 || { cat .logs/risk_and_cost.log; exit 1; }
	@$(PYTHON) src/prepare_date_ablation.py > .logs/prepare_date_ablation.log 2>&1 || { cat .logs/prepare_date_ablation.log; exit 1; }
	@echo 'Offline audit, error analysis, cost calculations and ablation preparation complete; details in .logs/'
