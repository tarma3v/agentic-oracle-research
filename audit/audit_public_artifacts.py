#!/usr/bin/env python3
"""Independent offline audit; stdlib only. Never imports reference code.

Run make fetch first to obtain hash-verified sources, including a canonical
JSON conversion of the revision-pinned HF Parquet. No network calls occur here.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "vendor" / "reference"
MODELS = ("gpt4o", "deepseek", "llama")
YES_NO = {"YES", "NO"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_commit():
    """Verify fetched files against the retained upstream provenance manifest."""
    provenance = json.loads((REPO.parent / "reference_source.json").read_text())
    for name, expected in provenance["sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"Reference snapshot differs from provenance: {name}")
    return provenance["commit"]


def row_key(row, columns):
    return tuple(str(row.get(key) or "").strip() for key in columns)


def normalized_text(value):
    return " ".join(str(value).replace("\\n", " ").split())


def semantic_key(row, columns):
    return tuple(normalized_text(row.get(key) or "") for key in columns)


def data_profile(rows, id_col="id"):
    counts = collections.Counter(r[id_col] for r in rows)
    fields = list(rows[0])
    exact = collections.Counter(row_key(r, fields) for r in rows)
    return {"rows": len(rows), "fields": fields, "unique_ids": len(counts), "id_equals_series_ticker": sum(r.get(id_col) == r.get("series_ticker") for r in rows), "ids_with_multiple_rows": sum(n > 1 for n in counts.values()), "rows_in_multi_id_groups": sum(n for n in counts.values() if n > 1), "duplicate_id_excess": len(rows) - len(counts), "id_multiplicity_histogram": dict(sorted(collections.Counter(counts.values()).items())), "exact_duplicate_excess": sum(n - 1 for n in exact.values()), "labels": dict(collections.Counter(r["ground_truth"].upper() for r in rows)), "categories": dict(collections.Counter(r["category"] for r in rows)), "null_or_empty_fields": {f: sum(r.get(f) in (None, "") for r in rows) for f in fields}, "close_time_range": [min(r["close_time"] for r in rows), max(r["close_time"] for r in rows)] if "close_time" in fields else None}


def prediction(row, prefix, field="decision"):
    return row.get(f"{prefix}_{field}", "").strip().upper()


def vote(row, weighted=False, suffix=""):
    scores = {"YES": 0.0, "NO": 0.0}
    for model in MODELS:
        name = model + suffix
        decision = prediction(row, name)
        if decision in YES_NO:
            scores[decision] += float(row.get(name + "_confidence") or 0) if weighted else 1
    # Matches analysis script. Runtime majority uses strict > instead.
    return "YES" if scores["YES"] >= scores["NO"] else "NO"


def metrics(rows, preds):
    valid = [(r, p) for r, p in zip(rows, preds) if p in YES_NO]
    correct = sum(p == r["ground_truth"].upper() for r, p in valid)
    return {"attempted": len(valid), "correct": correct, "accuracy": correct / len(valid) if valid else None, "all_row_accuracy": correct / len(rows)}


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    c = (p + z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [c-h, c+h]


def paired(rows, p1, p2):
    cells = collections.Counter()
    for r, a, b in zip(rows, p1, p2):
        if a in YES_NO and b in YES_NO:
            cells[(a == r["ground_truth"].upper(), b == r["ground_truth"].upper())] += 1
    return {"both_correct": cells[True, True], "a_only_correct": cells[True, False], "b_only_correct": cells[False, True], "both_wrong": cells[False, False], "valid_pairs": sum(cells.values())}


def match_occurrences(a, b, fields):
    def index(rows):
        counts, output = collections.Counter(), {}
        for i,row in enumerate(rows):
            key = row_key(row, fields)
            output[key + (counts[key],)] = i
            counts[key] += 1
        return output
    ai, bi = index(a), index(b)
    return [(i,bi[k]) for k,i in ai.items() if k in bi]


def calibration(rows, prefix):
    pairs = [(float(r[prefix + "_confidence"]), prediction(r, prefix) == r["ground_truth"].upper()) for r in rows if prediction(r, prefix) in YES_NO and r.get(prefix + "_confidence")]
    # Confidence is treated as P(chosen decision correct), not blindly as P(YES).
    bins = collections.defaultdict(list)
    for conf, correct in pairs:
        bins[min(9, int(conf * 10))].append((conf, correct))
    details = [{"bin": b, "n": len(v), "mean_confidence": statistics.mean(c for c, _ in v), "accuracy": statistics.mean(int(ok) for _, ok in v)} for b, v in sorted(bins.items())]
    return {"n": len(pairs), "mean_confidence": statistics.mean(c for c, _ in pairs), "accuracy": statistics.mean(int(ok) for _, ok in pairs), "decision_correctness_brier": statistics.mean((c-int(ok))**2 for c, ok in pairs), "ece_10_equal_width_bins": math.fsum(sorted(d["n"]*abs(d["mean_confidence"]-d["accuracy"]) for d in details))/len(pairs), "bins": details}


def canonical_numbers(value):
    """Discard insignificant platform-dependent floating point digits."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: canonical_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_numbers(item) for item in value]
    return value


def main():
    dfile = REPO / "data/kalshibench_v2_evaluation_Filtered.csv"
    afile = REPO / "results/Final_Architecture_A_results.csv"
    bfile = REPO / "results/Final_Architecture_B_results.csv"
    data, a, b = map(read_csv, (dfile, afile, bfile))
    provenance = json.loads((ROOT.parent / "sources_manifest.json").read_text())
    hf = provenance["kalshibench"]
    parquet = ROOT.parent / hf["path"]
    derived = ROOT.parent / hf["derived_json"]["path"]
    if digest(parquet) != hf["sha256"] or digest(derived) != hf["derived_json"]["sha256"]:
        raise ValueError("HF source integrity mismatch; inspect files and run make fetch")
    upstream = json.loads(derived.read_text())
    report = {
        "method": "Independent standard-library recomputation of saved predictions; no model/retrieval rerun",
        "repository_commit": reference_commit(),
        "hf_revision": hf["revision"],
        "hf_revision_pinned": True,
        "hf_rows_source": "Canonical JSON decoded from SHA-256-verified revision-pinned Parquet; no viewer endpoint",
        "numeric_serialization": "Floats rounded to 12 decimal places; ECE uses sorted math.fsum",
        "sha256": {str(f.relative_to(ROOT.parent)): digest(f) for f in (dfile, afile, bfile, parquet, derived)},
        "evaluation_data": data_profile(data), "upstream_data": data_profile(upstream["rows"]),
    }
    all_fields = list(data[0])
    upstream_keys = collections.Counter(row_key(r, all_fields) for r in upstream["rows"])
    subset_keys = collections.Counter(row_key(r, all_fields) for r in data)
    report["subset_membership"] = {"not_in_upstream_multiset": sum((subset_keys-upstream_keys).values()), "excluded_from_upstream": sum((upstream_keys-subset_keys).values())}
    report["result_alignment"] = {"a_rows": len(a), "b_rows": len(b), "a_unique_row_uid": len({r["row_uid"] for r in a}), "b_unique_row_uid": len({r["row_uid"] for r in b}), "row_uid_order_identical": [r["row_uid"] for r in a] == [r["row_uid"] for r in b], "input_ground_truth_question_criteria_match": all(d["id"] == x["question_id"] == y["question_id"] and d["ground_truth"].upper() == x["ground_truth"].upper() == y["ground_truth"].upper() and d["question"] == x["question_text"] == y["question_text"] and d["description"] == x["resolution_criteria"] == y["resolution_criteria"] for d,x,y in zip(data,a,b))}
    report["result_alignment"]["row_uid_intersection"] = len({r["row_uid"] for r in a} & {r["row_uid"] for r in b})
    semantic_fields = ["question_id", "question_text", "resolution_criteria", "ground_truth"]
    report["result_alignment"]["unique_semantic_keys"] = {tag: len({row_key(r,semantic_fields) for r in rows}) for tag,rows in (("A",a),("B",b))}
    normalized_input = [{"question_id":r["id"],"question_text":r["question"],"resolution_criteria":r["description"],"ground_truth":r["ground_truth"].upper()} for r in data]
    expected = collections.Counter(semantic_key(r,semantic_fields) for r in normalized_input)
    report["result_alignment"]["input_semantic_multiset_difference"] = {}
    for tag,rows in (("A",a),("B",b)):
        actual = collections.Counter(semantic_key(r,semantic_fields) for r in rows)
        report["result_alignment"]["input_semantic_multiset_difference"][tag] = {"extra":sum((actual-expected).values()),"missing":sum((expected-actual).values())}
    mismatches = []
    for i,(d,x,y) in enumerate(zip(data,a,b)):
        for tag,r in (("A",x),("B",y)):
            changed = [left for left,right in (("id","question_id"),("question","question_text"),("description","resolution_criteria"),("ground_truth","ground_truth")) if normalized_text(d[left]).upper() != normalized_text(r[right]).upper()]
            if changed:
                mismatches.append({"row_1based": i+1, "architecture":tag, "input_id":d["id"],"result_id":r["question_id"],"changed_fields":"|".join(changed), "input_question":d["question"], "result_question":r["question_text"], "input_criteria":d["description"], "result_criteria":r["resolution_criteria"], "input_truth":d["ground_truth"], "result_truth":r["ground_truth"]})
    report["result_alignment"]["position_mismatches_by_architecture"] = dict(collections.Counter(r["architecture"] for r in mismatches))
    report["result_alignment"]["normalization_for_input_comparison"] = "Replace literal escaped newline and collapse whitespace; normalize label case. Position comparison ignores letter case."
    if mismatches:
        with (ROOT/"input_result_mismatches.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(mismatches[0]));writer.writeheader();writer.writerows(mismatches)
    report["metrics"] = {}
    for model in MODELS:
        report["metrics"]["A_"+model] = metrics(a, [prediction(r, model) for r in a])
        for round_ in (1,2):
            report["metrics"][f"B_{model}_round{round_}"] = metrics(b,[prediction(r, f"{model}_round{round_}") for r in b])
    maj, weighted = [vote(r) for r in a], [vote(r, True) for r in a]
    afinal, bfinal = [r["final_decision"].upper() for r in a], [r["final_decision"].upper() for r in b]
    report["metrics"].update({"A_recomputed_majority": metrics(a,maj), "A_recomputed_weighted": metrics(a,weighted), "A_saved_final": metrics(a,afinal), "B_saved_final": metrics(b,bfinal)})
    bfirst = [vote(r,suffix="_round1") for r in b]
    report["metrics"]["B_round1_recomputed_majority"] = metrics(b,bfirst)
    report["paired"] = {"A_majority_vs_A_weighted": paired(a,maj,weighted)}
    report["paired"]["B_round1_majority_vs_B_final"] = paired(b,bfirst,bfinal)
    for name,fields in (("paper_text_truth_occurrence",["question_text","ground_truth"]),("strict_semantic_occurrence",semantic_fields),("row_uid_occurrence",["row_uid"])):
        matches = match_occurrences(a,b,fields)
        matched_a=[a[i] for i,j in matches]
        matched_bfinal=[bfinal[j] for i,j in matches]
        report["paired"][name] = {"join_fields":fields,"pairs":len(matches),"criteria_mismatch":sum(a[i]["resolution_criteria"] != b[j]["resolution_criteria"] for i,j in matches),"A_majority_vs_B_saved":paired(matched_a,[maj[i] for i,j in matches],matched_bfinal), "A_weighted_vs_B_saved":paired(matched_a,[weighted[i] for i,j in matches],matched_bfinal)}
        if name == "paper_text_truth_occurrence":
            mismatch_pairs = [{"a_csv_row_1based":i+2,"b_csv_row_1based":j+2,"a_row_uid":a[i]["row_uid"],"b_row_uid":b[j]["row_uid"],"question_text":a[i]["question_text"],"a_criteria":a[i]["resolution_criteria"],"b_criteria":b[j]["resolution_criteria"],"truth":a[i]["ground_truth"]} for i,j in matches if a[i]["resolution_criteria"] != b[j]["resolution_criteria"]]
            with (ROOT/"paper_join_criteria_mismatches.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(mismatch_pairs[0]));writer.writeheader();writer.writerows(mismatch_pairs)
    report["analysis_runtime_majority_difference_rows"] = sum(x != y for x,y in zip(maj,afinal))
    report["saved_is_correct_mismatch"] = {tag: sum((r["is_correct"].lower() == "true") != (r["final_decision"].upper() == r["ground_truth"].upper()) for r in rows) for tag,rows in (("A",a),("B",b))}
    report["calibration"] = {"A_"+m: calibration(a,m) for m in MODELS}
    report["calibration"].update({"B_"+m+"_round2": calibration(b,m+"_round2") for m in MODELS})
    report["model_error_or_invalid_decisions"] = {tag: {col: sum(bool(r[col]) for r in rows) for col in rows[0] if col.endswith("_error")} for tag,rows in (("A",a),("B",b))}
    conf = [statistics.mean(float(r[m+"_confidence"] or 0) for m in MODELS) for r in a]
    threshold = statistics.median(conf)
    selected = [i for i,r in enumerate(a) if len({prediction(r,m) for m in MODELS}) == 1 and conf[i] >= threshold]
    successes = sum(afinal[i] == a[i]["ground_truth"].upper() for i in selected)
    report["A_unanimous_high_confidence"] = {"threshold_same_sample_median": threshold, "selected": len(selected), "correct": successes, "errors": len(selected)-successes, "coverage": len(selected)/len(a), "accuracy": successes/len(selected), "accuracy_wilson_95": wilson(successes,len(selected)), "warning": "Same-sample selection, correlated groups; naive row-wise interval descriptive only"}
    selected_errors = [a[i] for i in selected if afinal[i] != a[i]["ground_truth"].upper()]
    with (ROOT/"high_confidence_unanimous_errors.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(a[0]));writer.writeheader();writer.writerows(selected_errors)
    report["grouped_metrics"] = {}
    for tag,rows,preds in (("A_majority",a,afinal),("A_weighted",a,weighted),("B",b,bfinal)):
        groups = collections.defaultdict(list)
        for row,pred in zip(rows,preds):
            groups[row["question_id"]].append(pred == row["ground_truth"].upper())
        multis = [v for v in groups.values() if len(v)>1]
        report["grouped_metrics"][tag] = {"groups": len(groups), "first_row_accuracy": statistics.mean(int(v[0]) for v in groups.values()), "macro_group_accuracy": statistics.mean(statistics.mean(map(int,v)) for v in groups.values()), "multi_groups": len(multis), "multi_all_correct": sum(all(v) for v in multis), "multi_all_wrong": sum(not any(v) for v in multis), "multi_mixed": sum(any(v) and not all(v) for v in multis)}
    report["latency_ms"] = {tag: {"observed":sum(bool(r["resolution_time_ms"]) for r in rows),"mean": statistics.mean(float(r["resolution_time_ms"]) for r in rows if r["resolution_time_ms"]), "median": statistics.median(float(r["resolution_time_ms"]) for r in rows if r["resolution_time_ms"])} for tag,rows in (("A",a),("B",b))}
    report["evidence_cache_files_committed"] = len(list((REPO/"cache/evidence").glob("*.json")))
    report = canonical_numbers(report)
    (ROOT / "audit_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    summary = json.dumps({key:report[key] for key in ("repository_commit","hf_revision","evaluation_data","subset_membership","result_alignment","metrics","paired","A_unanimous_high_confidence","grouped_metrics")},ensure_ascii=False,indent=2)
    (ROOT / "audit_console_output.json").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
