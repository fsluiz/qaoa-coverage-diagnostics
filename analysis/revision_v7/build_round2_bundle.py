"""Build a local scientific bundle; never publish or modify git state."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from manifest import PROJECT, HERE, MANIFEST, dependencies, verify

IDENTIFIER = "v7-round2-20261021"


def included_files():
    files = set(dependencies()) | {MANIFEST, HERE/"verification.json"}
    files.update(HERE.glob("baseline_before*.json"))
    for directory in ("data", "results", "scripts", "figures", "tests"):
        files.update(p for p in (PROJECT/directory).rglob("*") if p.is_file()
                     and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(PROJECT).parts))
    files.update(PROJECT.glob("requirements*.txt"))
    files.update((PROJECT/"paper").glob("*V7*.md"))
    files.update((PROJECT/"paper").glob("*V7*AUDITORIA.json"))
    files.update((PROJECT/"paper").glob("*V7*VERIFICACAO.json"))
    for suffix in ("*.py", "*.md", "*.json", "*.txt"):
        files.update((PROJECT/"paper/ibm_variance_campaign").glob(suffix))
    files.update((PROJECT/"analysis/ibm_variance_campaign/results").glob("*"))
    return sorted(p for p in files if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT/"paper/releases")
    parser.add_argument("--identifier", default=IDENTIFIER,
                        help="Version-specific archive identifier; existing archives are never replaced.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.identifier):
        parser.error("The identifier must be a single filename component using letters, digits, '.', '_' or '-'.")
    verify()
    check = json.loads((HERE/"verification.json").read_text())
    if check["status"] != "PASS" or not check["full_reanalysis"]:
        raise RuntimeError("A full successful verification is required before bundling.")
    if check.get("input_manifest_sha256") != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise RuntimeError("The verification must correspond to the current manifest.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir/f"{args.identifier}.tar.gz"
    files = included_files()
    manifest = dict(identifier=args.identifier, status="local bundle; not a public deposit",
                    files={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with tarfile.open(archive, "x:gz") as target:
        for path in files:
            target.add(path, arcname=str(Path(args.identifier)/path.relative_to(PROJECT)), recursive=False)
        registry = json.dumps(manifest, indent=2).encode()+b"\n"
        info = tarfile.TarInfo(f"{args.identifier}/RELEASE_FILES.json")
        info.size = len(registry)
        target.addfile(info, io.BytesIO(registry))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.out_dir/f"{archive.name}.sha256").write_text(f"{sha}  {archive.name}\n")
    manifest["archive_sha256"] = sha
    (args.out_dir/f"{args.identifier}-files.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(archive=str(archive), files=len(files), bytes=archive.stat().st_size,
                          sha256=sha), indent=2))


if __name__ == "__main__":
    main()
