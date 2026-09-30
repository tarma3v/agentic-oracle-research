"""Prepare paired prompts offline; never call models or execute reference code.

Default: 600 date-only requests using source highlights in both arms.
--design factorial: 1200 requests crossing date and evidence-body formatting.
Fetched source content and labels stay under ignored experiments/prepared/.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "vendor/reference"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.replace("\\n", " ").split())


def identity(row: dict) -> tuple:
    return tuple(normalized(row[k]) for k in
                 ("question_id", "question_text", "resolution_criteria"))


def csv_rows(path: Path) -> list:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def literal_prompts(path: Path) -> dict:
    """Read only string constants via AST; importing upstream would execute code."""
    wanted = {"ROUND_1_SYSTEM_PROMPT", "ROUND_1_USER_TEMPLATE"}
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != wanted or not all(isinstance(v, str) for v in result.values()):
        raise ValueError("Expected pinned Round 1 prompt string constants")
    return result


def format_sources(sources: list, body: str) -> str:
    if not sources:
        return "No evidence sources were retrieved."
    lines = []
    for index, source in enumerate(sources, 1):
        lines.extend([f"--- Source {index} ---", f"Title: {source['title']}",
                      f"URL: {source['url']}"])
        if source.get("published_date"):
            lines.append(f"Published: {source['published_date']}")
        # Reference B uses text only; reference A prefers highlights.
        content = source.get("text", "")
        if body == "highlights" and source.get("highlights"):
            content = "\n".join("- " + s for s in source["highlights"])
        lines.extend([content, ""])
    return "\n".join(lines)


def jsonl(path: Path, rows: list) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return sha(data.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=["date", "factorial"], default="date")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "vendor/reference_source.json").read_text())

    def verified(relative: str) -> Path:
        path = REPO / relative
        if sha(path.read_bytes()) != provenance["sha256"][relative]:
            raise ValueError("Reference hash mismatch: " + relative)
        return path

    by_uid = {}
    for tag in "AB":
        rows = csv_rows(verified(f"results/Final_Architecture_{tag}_results.csv"))
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row["row_uid"]].append(row)
        by_uid[tag] = grouped
    prompts = literal_prompts(verified("src/resolution/debate/prompts.py"))
    base_template = prompts["ROUND_1_USER_TEMPLATE"]
    marker = "RESOLUTION CRITERIA: {criteria}\n\n"
    if base_template.count(marker) != 1:
        raise ValueError("Pinned template insertion point changed")
    dated_template = base_template.replace(marker, marker + "Resolution Date: {date}\n\n")
    protocol = json.loads((ROOT / "experiments/date_ablation_protocol.json").read_text())
    requests, labels, packets = [], [], []
    body_modes = ["highlights"] if args.design == "date" else ["highlights", "text_only"]
    for uid in sorted(by_uid["A"].keys() & by_uid["B"].keys()):
        rows = by_uid["A"][uid] + by_uid["B"][uid]
        keys = {r["retrieval_mode"] + "__" + r["row_uid"] for r in rows}
        if len(keys) != 1:
            continue
        relative = "cache/evidence/" + next(iter(keys)) + ".json"
        if not (REPO / relative).exists():
            continue
        path = verified(relative)
        packet = json.loads(path.read_text())
        if not all(identity(row) == identity(packet) for row in rows):
            continue
        if len(by_uid["A"][uid]) != 1 or len(by_uid["B"][uid]) != 1:
            raise ValueError("Ambiguous row UID in selected sample")
        truths = {r["ground_truth"].upper() for r in rows}
        if len(truths) != 1 or not truths <= {"YES", "NO"}:
            raise ValueError("A/B evaluation labels differ in selected sample")
        if not packet.get("resolution_date"):
            raise ValueError("Missing resolution date in selected sample")
        labels.append({"row_uid": uid, "label": next(iter(truths))})
        packet_hash = sha(path.read_bytes())
        packets.append({"row_uid": uid, "packet": relative, "sha256": packet_hash})
        for body in body_modes:
            kwargs = {"question": packet["question_text"],
                      "criteria": packet["resolution_criteria"],
                      "exa_evidence": format_sources(packet["sources"], body),
                      "date": packet["resolution_date"]}
            without = base_template.format(**kwargs)
            with_date = dated_template.format(**kwargs)
            expected = without.replace("EVIDENCE PROVIDED from reputable sources:",
                                       "Resolution Date: " + kwargs["date"] +
                                       "\n\nEVIDENCE PROVIDED from reputable sources:", 1)
            if with_date != expected:
                raise ValueError("Date treatment changed more than its intended line")
            for model in protocol["model_slots"]:
                for condition, user_prompt in [("absent", without), ("present", with_date)]:
                    requests.append({
                        "request_id": f"{uid}:{model}:{body}:{condition}",
                        "row_uid": uid, "model_slot": model,
                        "evidence_body": body, "explicit_date": condition,
                        "packet_sha256": packet_hash,
                        "messages": [{"role": "system", "content": prompts["ROUND_1_SYSTEM_PROMPT"]},
                                     {"role": "user", "content": user_prompt}],
                    })
    if len(packets) != 100 or len(requests) != 100 * 3 * 2 * len(body_modes):
        raise ValueError("Pinned selection counts changed")
    # Fixed hash order avoids putting every treatment or model in one time block.
    requests.sort(key=lambda r: sha(r["request_id"].encode("utf-8")))
    assert len({r["request_id"] for r in requests}) == len(requests)
    assert all(not ({"label", "ground_truth", "is_correct", "final_decision"} & r.keys())
               for r in requests)
    out = ROOT / "experiments/prepared" / args.design
    out.mkdir(parents=True, exist_ok=True)
    request_hash = jsonl(out / "requests.jsonl", requests)
    label_hash = jsonl(out / "labels.jsonl", labels)
    manifest = {"status": "prepared_not_run", "design": args.design,
                "contracts": len(packets), "requests": len(requests),
                "requests_sha256": request_hash, "labels_sha256": label_hash,
                "protocol_sha256": sha((ROOT / "experiments/date_ablation_protocol.json").read_bytes()),
                "reference_commit": provenance["commit"], "packets": packets,
                "note": "Labels are evaluator-only. Filesystem separation alone is not an access boundary."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "design": args.design,
                      "contracts": len(packets), "requests": len(requests),
                      "requests_sha256": request_hash,
                      "directory": out.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
