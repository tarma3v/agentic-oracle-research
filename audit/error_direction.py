#!/usr/bin/env python3
"""Offline audit of error direction and retrospective confidence selection.

Uses only the standard library and saved upstream outputs; does not import or
execute upstream code, contact a service, or run any model. Output has no clock
fields and is deterministic. All risks are relative to the saved dataset label.
"""
import collections
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / "vendor" / "reference"
MODELS = ("gpt4o", "deepseek", "llama")
IDENTITY = ("question_id", "question_text", "resolution_criteria")
LABELS = {"YES", "NO"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized(value):
    return " ".join(str(value).replace("\\n", " ").split())


def identity(row):
    return tuple(normalized(row[field]) for field in IDENTITY)


def truth(row):
    label = row["ground_truth"].strip().upper()
    if label not in LABELS:
        raise ValueError("Non-binary ground truth: " + label)
    return label


def decision(row, prefix):
    return row[prefix + "_decision"].strip().upper()


def confidence(row, prefix):
    value = Decimal(row[prefix + "_confidence"])
    if not Decimal(0) <= value <= Decimal(1):
        raise ValueError("Confidence outside [0, 1]")
    return value


def majority(row, suffix="", weighted=False):
    scores = {"YES": Decimal(0), "NO": Decimal(0)}
    for model in MODELS:
        prefix = model + suffix
        prediction = decision(row, prefix)
        if prediction in LABELS:
            scores[prediction] += confidence(row, prefix) if weighted else 1
    # Historical analysis code resolves ties to YES. None of the three-agent
    # unweighted aggregate rows in these saved files has a tie.
    return "YES" if scores["YES"] >= scores["NO"] else "NO"


def divide(a, b):
    return a / b if b else None


def confusion(rows, predictions):
    assert len(rows) == len(predictions)
    counts = collections.Counter((truth(row), pred)
                                 for row, pred in zip(rows, predictions))
    tp, fn = counts["YES", "YES"], counts["YES", "NO"]
    fp, tn = counts["NO", "YES"], counts["NO", "NO"]
    valid = tp + fn + fp + tn
    recall_yes, recall_no = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        "n_rows": len(rows), "valid_binary_predictions": valid,
        "invalid_predictions": len(rows) - valid,
        "truth_yes": sum(truth(row) == "YES" for row in rows),
        "truth_no": sum(truth(row) == "NO" for row in rows),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "errors": fn + fp, "correct": tp + tn,
        "accuracy_all_rows": divide(tp + tn, len(rows)),
        "recall_yes": recall_yes, "recall_no": recall_no,
        "balanced_accuracy": ((recall_yes + recall_no) / 2
                              if recall_yes is not None and recall_no is not None else None),
        "false_no_share_of_binary_errors": divide(fn, fn + fp),
        "predicted_yes": tp + fp,
        "predicted_yes_fraction_all_rows": divide(tp + fp, len(rows)),
        "truth_yes_fraction_all_rows": divide(sum(truth(row) == "YES" for row in rows), len(rows)),
    }


def exact_mcnemar(a_only_correct, b_only_correct):
    n = a_only_correct + b_only_correct
    if not n:
        return 1.0
    # Exact two-sided binomial test under p=1/2; integer numerator first.
    numerator = 2 * sum(math.comb(n, k) for k in
                        range(min(a_only_correct, b_only_correct) + 1))
    return min(1.0, numerator / (2 ** n))


def paired(rows, left, right):
    assert len(rows) == len(left) == len(right)
    counts = collections.Counter()
    for row, a, b in zip(rows, left, right):
        if a not in LABELS or b not in LABELS:
            raise ValueError("Paired comparison requires binary predictions")
        counts[a == truth(row), b == truth(row)] += 1
    ac, bc = counts[True, False], counts[False, True]
    return {
        "n_pairs": len(rows), "both_correct": counts[True, True],
        "left_only_correct": ac, "right_only_correct": bc,
        "both_wrong": counts[False, False],
        "left_correct": counts[True, True] + ac,
        "right_correct": counts[True, True] + bc,
        "right_minus_left_correct": bc - ac,
        "exact_mcnemar_two_sided_p": exact_mcnemar(ac, bc),
    }


def occurrence_matches(a, b, fields):
    """Pair each repeated full key by its original CSV occurrence number."""
    def index(rows):
        seen, output = collections.Counter(), {}
        for i, row in enumerate(rows):
            key = tuple(normalized(row[field]) for field in fields)
            output[key + (seen[key],)] = i
            seen[key] += 1
        return output
    ai, bi = index(a), index(b)
    return [(i, bi[key]) for key, i in ai.items() if key in bi]


def matched_cache_candidates(a, b, repo=REPO):
    """Return available same-packet candidates, not evidence of past usage.

    Uses audit_cache_coverage.py's rule: shared row_uid, all expected filenames
    exist, one shared {retrieval_mode}__{row_uid}.json, all packet identities
    (question ID/text/criteria, whitespace normalized) agree. Fail on duplicate
    row_uid rather than silently choosing a pair. Source files currently have
    unique row_uid. Labels are checked separately and retained in each record.
    """
    by_uid = {}
    for tag, rows in (("A", a), ("B", b)):
        grouped = collections.defaultdict(list)
        for i, row in enumerate(rows):
            grouped[row["row_uid"]].append((i, row))
        by_uid[tag] = grouped
    output = []
    for uid in sorted(set(by_uid["A"]) & set(by_uid["B"])):
        a_rows, b_rows = by_uid["A"][uid], by_uid["B"][uid]
        both = a_rows + b_rows
        names = {row["retrieval_mode"] + "__" + row["row_uid"] + ".json"
                 for _, row in both}
        if len(names) != 1:
            continue
        filename = sorted(names)[0]
        path = repo / "cache" / "evidence" / filename
        if not path.exists():
            continue
        packet = json.loads(path.read_text(encoding="utf-8"))
        if not all(identity(row) == identity(packet) for _, row in both):
            continue
        if len(a_rows) != 1 or len(b_rows) != 1:
            raise ValueError("Nonunique cache-candidate row_uid: " + uid)
        ai, ar = a_rows[0]
        bi, br = b_rows[0]
        output.append({
            "row_uid": uid, "a_index_0based": ai, "b_index_0based": bi,
            "a_csv_line_1based": ai + 2, "b_csv_line_1based": bi + 2,
            "question_id": ar["question_id"],
            "a_ground_truth": truth(ar), "b_ground_truth": truth(br),
            "labels_agree": truth(ar) == truth(br),
            "packet_path_relative_to_reference": path.relative_to(repo).as_posix(),
            "packet_sha256": digest(path),
        })
    return output


def source_format_audit(candidates, repo=REPO):
    """Measure saved bodies under the two checked-in source-formatting rules.

    A Source.content_for_prompt prefers highlights and falls back to text;
    B DebateRunner._format_exa_sources appends text without that fallback.
    Both still print source title/URL and a publication date when available.
    This is a property of archived packets and code, not reconstructed LLM logs.
    """
    totals = collections.Counter()
    records = []
    for candidate in candidates:
        packet = json.loads((repo / candidate["packet_path_relative_to_reference"]).read_text(encoding="utf-8"))
        sources = packet.get("sources", [])
        counts = collections.Counter()
        for source in sources:
            text = source.get("text") or ""
            highlights = source.get("highlights") or []
            assert isinstance(text, str) and all(isinstance(item, str) for item in highlights)
            a_body = "\n".join("- " + item for item in highlights) if highlights else text
            counts["sources"] += 1
            counts["sources_empty_text"] += not text.strip()
            counts["sources_nonempty_highlights"] += any(item.strip() for item in highlights)
            counts["sources_empty_text_with_nonempty_highlights"] += (
                not text.strip() and any(item.strip() for item in highlights))
            counts["sources_with_nonempty_A_formatted_body"] += bool(a_body.strip())
            counts["sources_with_nonempty_B_formatted_body"] += bool(text.strip())
            counts["sources_with_title"] += bool(source.get("title"))
            counts["sources_with_url"] += bool(source.get("url"))
            counts["sources_with_publication_date"] += bool(source.get("published_date"))
        totals.update(counts)
        totals["packets"] += 1
        totals["packets_all_source_text_empty"] += bool(sources) and counts["sources_empty_text"] == len(sources)
        totals["packets_all_sources_have_highlights_but_empty_text"] += (
            bool(sources) and counts["sources_empty_text_with_nonempty_highlights"] == len(sources))
        totals["packets_with_resolution_date"] += bool(packet.get("resolution_date"))
        records.append({"row_uid": candidate["row_uid"], **dict(counts)})
    return {
        "scope": "The matching shared-cache candidate subset only",
        "A_source_body_rule": "src/retrieval/evidence.py: Source.content_for_prompt prefers highlights, otherwise text",
        "B_source_body_rule": "src/resolution/debate/runner.py: DebateRunner._format_exa_sources uses only source.text",
        "both_retain": "Title and URL, plus publication date if supplied",
        "interpretation": "Checked-in formatters produce different source bodies for these candidate packets. This is a competing confound to resolution-date omission; it is not proof of exact historical prompts or of causal impact.",
        "counts": dict(totals), "records": records,
    }


def gap(left, right):
    result = {"right_minus_left_" + key: right[key] - left[key]
              for key in ("FN", "FP", "errors", "correct", "truth_yes", "truth_no")}
    assert result["right_minus_left_errors"] == (
        result["right_minus_left_FN"] + result["right_minus_left_FP"])
    return result


def compare_groups(a, b, matches, ap, b1, bf):
    ar, br = [a[i] for i, _ in matches], [b[j] for _, j in matches]
    assert all(truth(x) == truth(y) for x, y in zip(ar, br))
    pa, p1, pf = ([ap[i] for i, _ in matches], [b1[j] for _, j in matches],
                  [bf[j] for _, j in matches])
    matrices = {"A_majority": confusion(ar, pa),
                "B_round1_majority": confusion(br, p1), "B_final": confusion(br, pf)}
    return {
        "n_pairs": len(matches), "confusion": matrices,
        "B_round1_minus_A": gap(matrices["A_majority"], matrices["B_round1_majority"]),
        "B_final_minus_A": gap(matrices["A_majority"], matrices["B_final"]),
        "paired": {"A_majority_vs_B_round1": paired(ar, pa, p1),
                   "A_majority_vs_B_final": paired(ar, pa, pf),
                   "B_round1_vs_B_final": paired(br, p1, pf)},
    }


def unanimous(row, suffix=""):
    predictions = {decision(row, model + suffix) for model in MODELS}
    return len(predictions) == 1 and predictions <= LABELS


def confidence_summary(rows, suffix=""):
    subsets = {"all_rows": rows,
               "unanimous_rows": [row for row in rows if unanimous(row, suffix)]}
    output = {}
    for name, selected in subsets.items():
        output[name] = {}
        for model in MODELS:
            values = [confidence(row, model + suffix) for row in selected]
            histogram = collections.Counter(values)
            output[name][model] = {
                "denominator_rows": len(values),
                "confidence_ge_0_95_count": sum(v >= Decimal("0.95") for v in values),
                "confidence_ge_0_95_fraction": divide(sum(v >= Decimal("0.95") for v in values), len(values)),
                "histogram": {str(value.normalize()): histogram[value] for value in sorted(histogram)},
            }
    return output


def error_count(rows, indices):
    return sum(majority(rows[i]) != truth(rows[i]) for i in indices)


def rank_selection(rows, pool, score, n):
    """Fixed coverage with explicit deterministic ordering and tie envelope."""
    scores = {i: score(rows[i]) for i in pool}
    ordered = sorted(pool, key=lambda i: (-scores[i], i))
    selected = ordered[:n]
    boundary = scores[selected[-1]]
    above = [i for i in pool if scores[i] > boundary]
    tied = [i for i in pool if scores[i] == boundary]
    take = n - len(above)
    assert 0 < take <= len(tied)
    above_errors, tied_errors = error_count(rows, above), error_count(rows, tied)
    minimum = above_errors + max(0, take - (len(tied) - tied_errors))
    maximum = above_errors + min(take, tied_errors)
    alternate = sorted(pool, key=lambda i: (-scores[i], rows[i]["row_uid"], i))[:n]
    selected_errors = error_count(rows, selected)
    assert minimum <= selected_errors <= maximum
    return {
        "accepted": n, "coverage_all_rows": n / len(rows),
        "coverage_unanimous_pool": n / len(pool),
        "errors_csv_order_tiebreak": selected_errors,
        "risk_csv_order_tiebreak": selected_errors / n,
        "errors_row_uid_lexicographic_tiebreak": error_count(rows, alternate),
        "boundary_score": float(boundary), "rows_strictly_above_boundary": len(above),
        "errors_strictly_above_boundary": above_errors,
        "rows_tied_at_boundary": len(tied), "errors_in_boundary_tie": tied_errors,
        "boundary_tie_slots_taken": take,
        "min_errors_over_all_boundary_tiebreaks": minimum,
        "max_errors_over_all_boundary_tiebreaks": maximum,
        "expected_errors_uniform_random_boundary_tiebreak": above_errors + take * tied_errors / len(tied),
        "accepted_row_uids_sha256": hashlib.sha256(
            ("\n".join(rows[i]["row_uid"] for i in sorted(selected)) + "\n").encode()).hexdigest(),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def main():
    paths = {tag: REPO / "results" / f"Final_Architecture_{tag}_results.csv"
             for tag in ("A", "B")}
    a, b = read_csv(paths["A"]), read_csv(paths["B"])
    ap, aw = [majority(row) for row in a], [majority(row, weighted=True) for row in a]
    b1, b2 = [majority(row, "_round1") for row in b], [majority(row, "_round2") for row in b]
    af, bf = [row["final_decision"].upper() for row in a], [row["final_decision"].upper() for row in b]
    systems = {"A_majority": (a, ap), "A_weighted": (a, aw), "A_saved_final": (a, af),
               "B_round1_majority": (b, b1), "B_round2_majority": (b, b2), "B_final": (b, bf)}
    for model in MODELS:
        systems["A_" + model] = a, [decision(row, model) for row in a]
        for round_ in (1, 2):
            systems[f"B_{model}_round{round_}"] = b, [decision(row, f"{model}_round{round_}") for row in b]
    matrices = {name: confusion(rows, predictions) for name, (rows, predictions) in systems.items()}
    strict_pairs = occurrence_matches(a, b, IDENTITY + ("ground_truth",))
    strict = compare_groups(a, b, strict_pairs, ap, b1, bf)
    amatched, bmatched = {i for i, _ in strict_pairs}, {j for _, j in strict_pairs}
    aun, bun = [i for i in range(len(a)) if i not in amatched], [i for i in range(len(b)) if i not in bmatched]
    unmatched = {
        "A_majority": confusion([a[i] for i in aun], [ap[i] for i in aun]),
        "B_round1_majority": confusion([b[j] for j in bun], [b1[j] for j in bun]),
        "B_final": confusion([b[j] for j in bun], [bf[j] for j in bun]),
    }
    raw_gap = {"B_round1_minus_A": gap(matrices["A_majority"], matrices["B_round1_majority"]),
               "B_final_minus_A": gap(matrices["A_majority"], matrices["B_final"]),
               "B_final_minus_B_round1": gap(matrices["B_round1_majority"], matrices["B_final"])}
    for bname in ("B_round1_majority", "B_final"):
        for field in ("FN", "FP", "errors", "correct"):
            assert matrices[bname][field] - matrices["A_majority"][field] == (
                strict["confusion"][bname][field] - strict["confusion"]["A_majority"][field]
                + unmatched[bname][field] - unmatched["A_majority"][field])
    candidates = matched_cache_candidates(a, b)
    cache_pairs = [(r["a_index_0based"], r["b_index_0based"]) for r in candidates if r["labels_agree"]]
    cache_comparison = compare_groups(a, b, cache_pairs, ap, b1, bf)
    pool = [i for i, row in enumerate(a) if unanimous(row)]
    mean_conf = lambda row: sum((confidence(row, model) for model in MODELS), Decimal(0)) / len(MODELS)
    threshold = statistics.median(mean_conf(row) for row in a)
    paper_selected = [i for i in pool if mean_conf(a[i]) >= threshold]
    n = len(paper_selected)
    signals = {"mean_raw_confidence": mean_conf}
    signals.update({model: (lambda row, model=model: confidence(row, model)) for model in MODELS})
    selections = {name: rank_selection(a, pool, score, n) for name, score in signals.items()}
    assert selections["mean_raw_confidence"]["errors_csv_order_tiebreak"] == error_count(a, paper_selected)
    mismatched_positions = [{"csv_line_1based": i + 2,
                             "a_row_uid": ar["row_uid"], "b_row_uid": br["row_uid"],
                             "a_truth": truth(ar), "b_truth": truth(br),
                             "identity_agrees": identity(ar) == identity(br)}
                            for i, (ar, br) in enumerate(zip(a, b)) if truth(ar) != truth(br)]
    categories = {}
    for name in ("A_majority", "B_round1_majority", "B_final"):
        rows, predictions = systems[name]
        categories[name] = {}
        for category in sorted({row["category"] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row["category"] == category]
            categories[name][category] = confusion([rows[i] for i in indices], [predictions[i] for i in indices])
    report = {
        "schema_version": 1,
        "method": {
            "data": "Saved final architecture CSV predictions; independent offline recomputation, no model reruns",
            "ground_truth": "Each saved CSV's own ground_truth; strict paired comparisons require matching labels",
            "positive_label": "YES", "FN": "truth YES, prediction NO", "FP": "truth NO, prediction YES",
            "model_columns": "gpt4o/deepseek/llama are historical CSV column names, not independently verified invocation logs",
            "confidence": "Reported probability of own selected decision being correct; not probability of YES",
            "normalization": "Replace literal backslash-n with spaces and collapse whitespace; label case uppercased for metrics",
            "strict_pairing": "Normalized question_id, question_text, resolution_criteria, ground_truth plus within-key occurrence order",
            "cache_pairing": "Same shared row_uid and expected packet filename; normalized ID/text/criteria match for every row; labels checked separately",
            "cache_limitation": "Available shared candidate packet does not prove historical calls used the same evidence bytes; no immutable run-to-packet binding exists in these CSVs",
            "p_values": "Exact two-sided McNemar/binomial; descriptive row-wise calculation, not corrected for repeated event/series clustering or post-hoc selection",
            "selection": "Retrospective same-sample ranking of A unanimous rows at the saved mean-confidence rule's coverage; not a held-out calibration experiment",
            "ties": "Default descending exact-decimal score, original CSV row order; report alternate UID order, best/worst attainable boundary choices and uniform-random expected error count",
            "limitations": [
                "Raw A/B gap is an arithmetic difference across nonidentical row content and labels, not a paired causal effect",
                "Strict matching controls saved contract content and label, not prompts, retrieval or execution state",
                "The cached subset is selected by artifact availability and is not a representative random sample",
                "Confidence scales and tie concentration differ; these observations alone do not prove no possible benefit from averaging",
                "Evidence absence causing false NO and missing resolution date causing the A/B gap remain untested mechanisms",
                "Category counts do not establish causes; any assertion about unobserved crypto prices needs packet-level review",
            ],
            "float_serialization": "Round to 12 decimal places; no runtime timestamps",
        },
        "source_sha256": {path.relative_to(REPO).as_posix(): digest(path) for path in
                          list(paths.values()) + [REPO / "src/retrieval/evidence.py",
                                                  REPO / "src/resolution/debate/runner.py"]},
        "confusion_matrices": matrices,
        "saved_vs_recomputed_aggregate_disagreements": {
            "A_saved_vs_majority": sum(x != y for x, y in zip(af, ap)),
            "B_saved_vs_round2_majority": sum(x != y for x, y in zip(bf, b2)),
        },
        "gap_decomposition": {
            "raw_own_label_unpaired": raw_gap,
            "position_label_mismatch_count": len(mismatched_positions),
            "position_label_mismatches": mismatched_positions,
            "strict_matched": strict,
            "unmatched_own_label_matrices": unmatched,
            "unmatched_B_round1_minus_A": gap(unmatched["A_majority"], unmatched["B_round1_majority"]),
            "unmatched_B_final_minus_A": gap(unmatched["A_majority"], unmatched["B_final"]),
            "accounting_identity": "Raw B-minus-A error count = strict-matched error difference + unmatched-row error difference; this identity is asserted in code",
        },
        "B_round1_vs_B_final_all_rows": paired(b, b1, bf),
        "shared_cache_candidates": {
            "available_candidate_pairs": len(candidates),
            "matching_label_pairs": len(cache_pairs),
            "different_label_pairs": len(candidates) - len(cache_pairs),
            "comparison": cache_comparison,
            "records": candidates,
        },
        "candidate_source_format_audit": source_format_audit(candidates),
        "confidence_distributions": {"A": confidence_summary(a), "B_round1": confidence_summary(b, "_round1"),
                                     "B_round2": confidence_summary(b, "_round2")},
        "escalation_equal_coverage": {
            "architecture": "A", "all_rows": len(a), "eligible_unanimous_rows": len(pool),
            "mean_confidence_median_over_all_rows": float(threshold),
            "saved_rule_accepted": n, "saved_rule_errors": error_count(a, paper_selected),
            "signals": selections,
        },
        "category_confusion_matrices": categories,
    }
    report = rounded(report)
    output = ROOT / "error_direction.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": "audit/error_direction.json", "sha256": digest(output),
                      "aggregate_confusion": {name: matrices[name] for name in ("A_majority", "B_round1_majority", "B_final")},
                      "cache_pairs": len(cache_pairs), "selection": report["escalation_equal_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
