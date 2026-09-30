#!/usr/bin/env python3
"""Fetch pinned research inputs; never import or execute the reference project.

Only the Parquet-to-JSON conversion needs pyarrow (isolated by make fetch).
Integrity verification and the later offline audits use the standard library.
Existing files that fail a recorded hash are rejected, never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_or_missing(path, expected):
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f"Integrity mismatch: {path}; inspect/move this local file before fetching again")
    return True


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe manifest path: {name}")
    return path


def download(url, destination):
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS download URLs are accepted")
    # Use the platform curl trust store (Python.org macOS installs can lack CA
    # configuration). Certificate verification stays enabled on every redirect.
    subprocess.run([
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--proto", "=https", "--proto-redir", "=https", "--retry", "3",
        "--connect-timeout", "30", "--max-time", "180",
        "--output", str(destination), url,
    ], check=True)


def reference(manifest, verify_only):
    entry = manifest["reference"]
    provenance = json.loads((ROOT / entry["provenance_manifest"]).read_text())
    expected = provenance["sha256"]
    destination = ROOT / entry["local_path"]
    for name in expected:
        safe_relative(name)
    missing = [name for name, sha in expected.items()
               if not valid_or_missing(destination / name, sha)]
    if missing and verify_only:
        raise ValueError(f"Missing {len(missing)} reference files; run make fetch")
    if missing:
        # A codeload tar is only a transport: every extracted byte is verified
        # against the pre-existing original per-file manifest before installation.
        if provenance["commit"] not in entry["archive_url"]:
            raise ValueError("Reference archive URL does not name the pinned commit")
        with tempfile.TemporaryDirectory(prefix="oracle-reference-") as temporary:
            temporary = Path(temporary)
            archive = temporary / "source.tar.gz"
            download(entry["archive_url"], archive)
            extracted = temporary / "verified"
            with tarfile.open(archive, "r:gz") as bundle:
                members = {}
                for member in bundle.getmembers():
                    if member.isdir():
                        continue
                    name = PurePosixPath(*PurePosixPath(member.name).parts[1:]).as_posix()
                    safe_relative(name)
                    if not member.isfile() or name in members:
                        raise ValueError(f"Non-regular or duplicate archive entry: {member.name}")
                    members[name] = member
                if set(members) != set(expected):
                    raise ValueError("Reference archive file list differs from the pinned manifest")
                for name, sha in expected.items():
                    target = extracted / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(members[name]) as source, target.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    valid_or_missing(target, sha)
            for name in missing:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(extracted / name, target)
    print(f"Reference: {len(expected)} SHA-256 checks passed at {provenance['commit']}")


def huggingface(manifest, verify_only):
    entry = manifest["kalshibench"]
    parquet = ROOT / entry["path"]
    derived = ROOT / entry["derived_json"]["path"]
    if entry["revision"] not in entry["url"]:
        raise ValueError("HF download URL does not name the pinned revision")
    if not valid_or_missing(parquet, entry["sha256"]):
        if verify_only:
            raise ValueError("Missing pinned Parquet; run make fetch")
        parquet.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="oracle-hf-") as temporary:
            candidate = Path(temporary) / "source.parquet"
            download(entry["url"], candidate)
            valid_or_missing(candidate, entry["sha256"])
            shutil.copyfile(candidate, parquet)
    if not valid_or_missing(derived, entry["derived_json"]["sha256"]):
        if verify_only:
            raise ValueError("Missing canonical HF JSON; run make fetch")
        import pyarrow.parquet as pq  # Fetch-only dependency; never used by audit.
        rows = pq.read_table(parquet).to_pylist()
        if len(rows) != entry["derived_json"]["rows"]:
            raise ValueError("Unexpected Parquet row count")
        payload = {"revision": entry["revision"], "source_sha256": entry["sha256"], "rows": rows}
        serialized = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if hashlib.sha256(serialized).hexdigest() != entry["derived_json"]["sha256"]:
            raise ValueError("Decoded HF rows differ from the expected canonical JSON")
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_bytes(serialized)
    print(f"KalshiBench: pinned Parquet and canonical {entry['derived_json']['rows']}-row JSON hashes passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Offline: check all required inputs without downloading")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sources_manifest.json").read_text())
    reference(manifest, args.verify_only)
    huggingface(manifest, args.verify_only)


if __name__ == "__main__":
    main()
