#!/usr/bin/env python3
"""Read-only Kalshi public-data smoke test, using only the Python standard library.

This is a recent convenience sample, NOT a benchmark or an as-of backtest.
The input allowlist removes outcome/price fields, but historical versions of rules,
timing metadata, and source documents are not reconstructed by this collector.
Only public GET requests are made; no API keys, paid APIs, or trading calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
PROJECT = Path(__file__).resolve().parents[1]
INPUT_MARKET_FIELDS = (
    "ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title",
    "market_type", "rules_primary", "rules_secondary", "strike_type",
    "floor_strike", "cap_strike", "functional_strike", "early_close_condition",
    "can_close_early", "created_time", "open_time", "close_time",
    "expected_expiration_time", "expiration_time", "latest_expiration_time",
    "occurrence_datetime",
)
FORBIDDEN_INPUT_KEYS = {
    "result", "status", "expiration_value", "updated_time", "volume_fp",
    "volume_24h_fp", "open_interest_fp", "liquidity_dollars", "notional_value_dollars",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_safe_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = key.lower()
            if key in FORBIDDEN_INPUT_KEYS or "price" in lower or "settlement" in lower:
                raise ValueError(f"Outcome or price field leaked into inputs: {key}")
            assert_safe_keys(child)
    elif isinstance(value, list):
        for child in value:
            assert_safe_keys(child)


class PublicReader:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.context = ssl.create_default_context()
        # Some macOS Python installations have no default CA bundle. Keep TLS
        # verification enabled and load the operating system's bundle if present.
        system_ca = Path("/etc/ssl/cert.pem")
        if system_ca.is_file():
            self.context.load_verify_locations(cafile=str(system_ca))

    def get(self, endpoint: str, filename: str, params: dict | None = None) -> dict:
        url = BASE_URL + endpoint
        if params:
            url += "?" + urlencode(params)
        last_error = None
        for attempt in range(3):
            try:
                request = Request(url, headers={"Accept": "application/json", "User-Agent": "agentic-oracle-research-smoke-test/1.0"}, method="GET")
                with urlopen(request, context=self.context, timeout=30) as response:
                    body = response.read()
                    metadata = {
                        "url": url, "method": "GET", "retrieved_at": now(),
                        "http_status": response.status,
                        "response_date": response.headers.get("Date"),
                        "content_type": response.headers.get("Content-Type"),
                        "raw_file": str(self.raw_dir / filename),
                        "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                    }
                data = json.loads(body)
                (self.raw_dir / filename).write_bytes(body)
                with self.lock:
                    self.requests.append(metadata)
                return data
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    break
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {last_error}")


def source_pairs(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [{"name": row.get("name"), "url": row.get("url")} for row in value
            if isinstance(row, dict) and row.get("url")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="1..100 recent markets")
    parser.add_argument("--max-series", type=int, default=5, help="0..5 full series metadata requests")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data" / "kalshi")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "data_smoke_test.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or not 0 <= args.max_series <= 5:
        parser.error("--limit must be 1..100 and --max-series must be 0..5")

    started_at = now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    data_dir = args.data_dir.resolve()
    reader = PublicReader(data_dir / "raw" / run_id)
    cutoff = reader.get("/historical/cutoff", "cutoff.json")
    market_response = reader.get("/markets", "markets.json", {
        "status": "settled", "limit": args.limit, "mve_filter": "exclude",
    })
    market_rows = market_response.get("markets", [])
    if not isinstance(market_rows, list):
        raise ValueError("Unexpected API shape: markets must be a list")

    selected = []
    excluded = Counter()
    seen = set()
    for row in market_rows:
        ticker = row.get("ticker")
        if not ticker or ticker in seen:
            excluded["missing_or_duplicate_ticker"] += 1
            continue
        seen.add(ticker)
        if row.get("mve_collection_ticker") or row.get("mve_selected_legs") or ticker.startswith("KXMVE"):
            excluded["combo"] += 1
            continue
        if row.get("status") not in ("settled", "finalized") or not row.get("settlement_ts"):
            excluded["not_confirmed_settled"] += 1
            continue
        selected.append(row)
    if not selected:
        raise RuntimeError("No settled non-combo markets returned; raw responses retained")

    event_ids = sorted({row["event_ticker"] for row in selected if row.get("event_ticker")})
    events = {}
    request_errors = []
    # Event responses provide the actual series link; do not infer it from ticker text.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(reader.get, "/events/" + quote(event_id, safe=""),
                               f"event_{index:03d}.json", {"with_nested_markets": "false"}): event_id
                   for index, event_id in enumerate(event_ids)}
        for future in as_completed(futures):
            event_id = futures[future]
            try:
                events[event_id] = future.result().get("event", {})
            except Exception as exc:
                request_errors.append({"kind": "event", "id": event_id, "error": str(exc)})

    series_counts = Counter(events.get(row.get("event_ticker"), {}).get("series_ticker") for row in selected)
    series_counts.pop(None, None)
    selected_series = [ticker for ticker, _ in sorted(series_counts.items(), key=lambda item: (-item[1], item[0]))[:args.max_series]]
    series = {}
    for index, ticker in enumerate(selected_series):
        try:
            series[ticker] = reader.get("/series/" + quote(ticker, safe=""), f"series_{index:02d}.json").get("series", {})
        except Exception as exc:
            request_errors.append({"kind": "series", "id": ticker, "error": str(exc)})

    inputs = []
    labels = []
    for row in selected:
        event = events.get(row.get("event_ticker"), {})
        series_id = event.get("series_ticker")
        series_row = series.get(series_id, {})
        event_sources = source_pairs(event.get("settlement_sources"))
        series_sources = source_pairs(series_row.get("settlement_sources"))
        source_origin = "event_metadata" if event_sources else "series_metadata" if series_sources else None
        safe = {key: row.get(key) for key in INPUT_MARKET_FIELDS}
        safe.update({
            "series_ticker": series_id,
            "event_title": event.get("title"), "category": event.get("category"),
            "resolution_sources": event_sources or series_sources,
            "resolution_sources_origin": source_origin,
            "contract_url": series_row.get("contract_url"),
            "contract_terms_url": series_row.get("contract_terms_url"),
            "series_metadata_fetched": series_id in series,
        })
        assert_safe_keys(safe)
        inputs.append(safe)
        labels.append({key: row.get(key) for key in (
            "ticker", "event_ticker", "result", "status", "settlement_ts",
            "settlement_value_dollars", "expiration_value",
        )})

    write_jsonl(data_dir / "inputs.jsonl", inputs)
    write_jsonl(data_dir / "labels.jsonl", labels)
    event_counts = Counter(row.get("event_ticker") for row in inputs)
    duplicate_groups = {key: value for key, value in sorted(event_counts.items()) if value > 1}
    n = len(inputs)
    complete_fields = ("ticker", "event_ticker", "series_ticker", "title", "rules_primary",
                       "rules_secondary", "close_time", "resolution_sources", "contract_terms_url")
    completeness = {field: {"present": sum(bool(row.get(field)) for row in inputs), "total": n}
                    for field in complete_fields}
    missing_series_ids = sorted(set(series_counts) - set(series))
    missing_series_markets = sum(not row["series_metadata_fetched"] for row in inputs)
    warnings = [
        "Convenience sample of recent API results; this is a data-access smoke test, not a benchmark.",
        "No historical as-of rule versions or dated evidence snapshots were reconstructed.",
        "close_time and other metadata are retrospective snapshots and can encode operational outcome information; whitelist alone does not make an as-of backtest safe.",
        "Source URLs identify designated sources; their factual evidence has not been fetched or verified.",
        "Raw API responses and labels contain outcomes. Do not expose them to a resolver under evaluation.",
        "Do not randomly split market rows: multiple strikes share the same underlying event.",
    ]
    if missing_series_markets:
        warnings.append(f"Full series metadata was limited to {args.max_series} series: absent for {missing_series_markets}/{n} markets across {len(missing_series_ids)} linked series. Event-level sources may still be present.")
    if request_errors:
        warnings.append(f"{len(request_errors)} metadata requests failed; see request_errors.")
    summary = {
        "run_id": run_id, "sample_kind": "convenience_sample_smoke_test_not_benchmark",
        "started_at": started_at, "completed_at": now(),
        "requested_limit": args.limit, "raw_market_count": len(market_rows), "market_count": n,
        "excluded_counts": dict(excluded), "pagination_followed": False,
        "additional_page_available": bool(market_response.get("cursor")),
        "api_returned_status_counts": dict(Counter(row.get("status") for row in selected)),
        "label_counts": dict(Counter(row.get("result") for row in labels)),
        "category_counts": dict(Counter(row.get("category") or "MISSING" for row in inputs)),
        "unique_events": len(event_counts), "duplicate_event_group_count": len(duplicate_groups),
        "markets_in_duplicate_event_groups": sum(duplicate_groups.values()),
        "excess_rows_beyond_one_per_event": n - len(event_counts),
        "duplicate_event_groups": duplicate_groups,
        "completeness": completeness,
        "series_linked_count": len(series_counts), "series_fetched_count": len(series),
        "series_fetch_limit": args.max_series, "missing_full_series_metadata_market_count": missing_series_markets,
        "missing_full_series_metadata_ids": missing_series_ids,
        "missing_resolution_sources_count": sum(not row["resolution_sources"] for row in inputs),
        "forbidden_input_key_check": "passed",
        "cutoff": cutoff, "request_errors": request_errors, "warnings": warnings,
        "files": {"inputs": str(data_dir / "inputs.jsonl"), "labels": str(data_dir / "labels.jsonl"),
                  "manifest": str(data_dir / "manifest.json"), "raw_dir": str(reader.raw_dir)},
    }
    write_json(args.results.resolve(), summary)
    manifest = {
        "run_id": run_id, "sample_kind": summary["sample_kind"],
        "started_at": started_at, "completed_at": summary["completed_at"],
        "source": "Kalshi public API", "api_base_url": BASE_URL,
        "request_policy": "GET only; no credentials; no trading; no paid API",
        "sampling": {"method": "first API page; server default ordering; convenience sample",
                     "params": {"status": "settled", "limit": args.limit, "mve_filter": "exclude"},
                     "max_full_series_requests": args.max_series},
        "cutoff": cutoff,
        "historical_api_note": "Older settled markets are in /historical/markets. This smoke test intentionally samples only the live endpoint's recent settled tier.",
        "input_allowlist": list(INPUT_MARKET_FIELDS),
        "input_metadata_note": "Additional fields are explicit event/series metadata and designated source URLs. No factual evidence documents were downloaded.",
        "warnings": warnings, "request_errors": request_errors,
        "requests": sorted(reader.requests, key=lambda row: row["url"]),
        "artifacts": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
                      for path in (data_dir / "inputs.jsonl", data_dir / "labels.jsonl", args.results.resolve(), Path(__file__).resolve())},
    }
    write_json(data_dir / "manifest.json", manifest)
    write_json(reader.raw_dir / "manifest.json", manifest)
    print(json.dumps({key: summary[key] for key in (
        "run_id", "market_count", "unique_events", "duplicate_event_group_count", "category_counts",
        "label_counts", "series_fetched_count", "missing_full_series_metadata_market_count",
        "missing_resolution_sources_count", "forbidden_input_key_check",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
