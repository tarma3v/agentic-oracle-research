#!/usr/bin/env python3
"""Measure saved evidence-cache coverage without executing reference code.

Matches establish available candidate packets, not which bytes a historical
model invocation actually consumed. Only the Python standard library is used.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDORED = ROOT.parent / "vendor/reference"
REPO = VENDORED if VENDORED.exists() else ROOT / "reference-repo"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(value):
    return " ".join(str(value).replace("\\n", " ").split())


def identity(row):
    return tuple(normalized(row[field]) for field in
                 ("question_id", "question_text", "resolution_criteria"))


def expected_cache_key(row):
    return row["retrieval_mode"] + "__" + row["row_uid"]


def main():
    paths = sorted((REPO / "cache/evidence").glob("*.json"))
    if not paths:
        raise SystemExit("No evidence packets found in " + str(REPO))
    packets = {path.stem: json.loads(path.read_text()) for path in paths}
    semantic = collections.defaultdict(list)
    for key, packet in packets.items():
        semantic[identity(packet)].append(key)
    packet_ids = {packet["question_id"] for packet in packets.values()}
    cache_manifest = [
        {"path": path.relative_to(REPO).as_posix(),
         "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in paths
    ]
    canonical_manifest = json.dumps(cache_manifest, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")
    report = {
        "declared_reference_commit": "f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c",
        "method": {
            "unit": "CSV rows; cross-architecture section uses unique row_uid values",
            "expected_filename": "{retrieval_mode}__{row_uid}.json",
            "identity_fields": ["question_id", "question_text", "resolution_criteria"],
            "normalization": "Replace literal backslash-n with spaces, then collapse whitespace",
            "excluded_identity_fields": ["ground_truth", "resolution_date"],
            "limitation": (
                "A matching filename and identity identify an available candidate packet. "
                "Final CSVs do not bind each historical model call to an immutable packet "
                "hash; resolution_date is not present in those CSVs. Neither filename "
                "nor semantic matches prove which packet was actually consumed or "
                "historical availability of source content. ID-only matches are weaker."
            ),
        },
        "cache": {
            "json_files": len(paths),
            "highlights_row_uid_files": sum(key.startswith("highlights__") for key in packets),
            "legacy_id_named_files": sum(not key.startswith("highlights__") for key in packets),
            "unique_question_ids": len(packet_ids),
            "unique_normalized_identities": len(semantic),
            "packets_without_sources": sum(not p.get("sources") for p in packets.values()),
            "retrieval_timestamp_range": [
                min(p["retrieval_timestamp"] for p in packets.values()),
                max(p["retrieval_timestamp"] for p in packets.values()),
            ],
            "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
            "manifest_hash_encoding": "UTF-8 JSON, sort_keys=True, separators=(',', ':'), no newline",
            "files": cache_manifest,
        },
        "sha256": {},
        "architectures": {},
    }
    rows_by_tag = {}
    for tag in ("A", "B"):
        path = REPO / "results" / f"Final_Architecture_{tag}_results.csv"
        rows = rows_by_tag[tag] = read_csv(path)
        report["sha256"][path.relative_to(REPO).as_posix()] = digest(path)
        filename_rows = [row for row in rows if expected_cache_key(row) in packets]
        consistent = [row for row in filename_rows
                      if identity(row) == identity(packets[expected_cache_key(row)])]
        report["architectures"][tag] = {
            "rows": len(rows),
            "unique_row_uid": len({row["row_uid"] for row in rows}),
            "retrieval_modes": dict(sorted(collections.Counter(row["retrieval_mode"] for row in rows).items())),
            "expected_filename_exists": len(filename_rows),
            "expected_filename_and_normalized_identity_match": len(consistent),
            "expected_filename_but_identity_mismatch": len(filename_rows) - len(consistent),
            "any_packet_normalized_identity_match": sum(identity(row) in semantic for row in rows),
            "any_packet_same_question_id_only": sum(row["question_id"] in packet_ids for row in rows),
        }
    by_uid = {}
    for tag, rows in rows_by_tag.items():
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row["row_uid"]].append(row)
        by_uid[tag] = grouped
    common = set(by_uid["A"]) & set(by_uid["B"])
    both_cache, same_cache_and_identity = 0, 0
    for uid in common:
        a_rows, b_rows = by_uid["A"][uid], by_uid["B"][uid]
        if all(expected_cache_key(row) in packets for row in a_rows + b_rows):
            both_cache += 1
            keys = {expected_cache_key(row) for row in a_rows + b_rows}
            if len(keys) == 1 and all(identity(row) == identity(packets[expected_cache_key(row)])
                                      for row in a_rows + b_rows):
                same_cache_and_identity += 1
    report["cross_architecture_unique_row_uid"] = {
        "intersection": len(common),
        "all_expected_cache_files_exist_for_both": both_cache,
        "same_expected_cache_and_all_normalized_identities_match": same_cache_and_identity,
        "duplicate_uid_handling": "Every A/B row with a shared UID must satisfy the condition; no keep-first join",
    }
    error_path = ROOT / "high_confidence_unanimous_errors.csv"
    errors = read_csv(error_path)
    report["sha256"]["audit/high_confidence_unanimous_errors.csv"] = digest(error_path)
    report["high_confidence_unanimous_errors"] = {
        "rows": len(errors),
        "any_packet_normalized_identity_match": sum(identity(row) in semantic for row in errors),
        "expected_filename_and_normalized_identity_match": sum(
            expected_cache_key(row) in packets
            and identity(row) == identity(packets[expected_cache_key(row)]) for row in errors),
        "manual_cause_analysis_performed": False,
        "candidates": [{
            "row_uid": row["row_uid"], "question_id": row["question_id"],
            "expected_cache_filename": expected_cache_key(row) + ".json",
            "semantic_candidate_filenames": [key + ".json" for key in sorted(semantic.get(identity(row), []))],
        } for row in errors],
    }
    output = ROOT / "cache_coverage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cache_json_files": len(paths), "architectures": report["architectures"],
                      "cross_architecture": report["cross_architecture_unique_row_uid"],
                      "high_confidence_errors_with_candidates": report["high_confidence_unanimous_errors"]["any_packet_normalized_identity_match"],
                      "cache_manifest_sha256": report["cache"]["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
