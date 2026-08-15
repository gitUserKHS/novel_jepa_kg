"""Package a trained JEPA version for a teammate, and install one you were sent.

A clone of this repository carries no model and no data: `data/`, `checkpoints/`,
and `artifacts/versions/` are all ignored by git. One promoted version is about
40 MB, so it travels fine as a release asset or a file share -- there is no need
for a model host.

    python scripts/share_artifact.py export                    # active version
    python scripts/share_artifact.py export --version <id> -o out.zip
    python scripts/share_artifact.py import out.zip            # verify + install
    python scripts/share_artifact.py import out.zip --activate

Import verifies every file against the SHA-256 fingerprints in the manifest
before installing, so a truncated download fails loudly instead of producing a
model that quietly generates nonsense.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.artifacts import VERSION_RE, sha256_file  # noqa: E402
from src.utils.config import AppConfig, load_config  # noqa: E402
from src.utils.paths import resolve_path  # noqa: E402

MANIFEST_NAME = "manifest.json"


def default_bundle_name(version: str) -> str:
    """Name a bundle the way the published release assets are named.

    A bare `<version>.zip` says nothing about what it holds once it is sitting in
    a downloads folder next to other files.
    """
    return f"jepa-model-{version}.zip"


def _config(path: str) -> AppConfig:
    return load_config(Path(path))


def _versions_dir(config: AppConfig) -> Path:
    return resolve_path(config, config.consumer.versions_dir)


def _active_version(config: AppConfig) -> str:
    active = resolve_path(config, config.consumer.active_manifest_path)
    if not active.exists():
        raise SystemExit("No active model. Promote one first, or pass --version.")
    manifest = json.loads(active.read_text(encoding="utf-8"))
    return str(manifest.get("version", "")).strip()


def export_version(config: AppConfig, version: str, output: Path) -> Path:
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"Invalid version id: {version}")
    source = _versions_dir(config) / version
    if not source.is_dir():
        raise SystemExit(f"Version directory not found: {source}")
    manifest_path = source / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"Version has no {MANIFEST_NAME}: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source.iterdir() if path.is_file())
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            # Store under the version id so an import knows where it belongs
            # without trusting any absolute path recorded at build time.
            bundle.write(path, arcname=f"{version}/{path.name}")
    # Report the archive size, since that is what actually gets uploaded.
    print(f"Exported {version} ({len(files)} files, {output.stat().st_size / 1e6:.1f} MB) to {output}")
    return output


def _bundle_version(bundle: zipfile.ZipFile) -> str:
    roots = {Path(name).parts[0] for name in bundle.namelist() if name.strip()}
    if len(roots) != 1:
        raise SystemExit("Bundle must contain exactly one version directory.")
    version = roots.pop()
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"Bundle version id is invalid: {version}")
    return version


def import_bundle(config: AppConfig, archive: Path, *, activate: bool) -> str:
    if not archive.exists():
        raise SystemExit(f"Bundle not found: {archive}")
    target_root = _versions_dir(config)
    target_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as bundle:
        version = _bundle_version(bundle)
        destination = target_root / version
        if destination.exists():
            raise SystemExit(
                f"Version {version} is already installed. Delete {destination} to reinstall."
            )
        staging = target_root / f".incoming-{version}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            for name in bundle.namelist():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit(f"Refusing unsafe path in bundle: {name}")
                if name.endswith("/"):
                    continue
                extracted = staging / relative.name
                with bundle.open(name) as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)

            manifest_path = staging / MANIFEST_NAME
            if not manifest_path.exists():
                raise SystemExit(f"Bundle has no {MANIFEST_NAME}.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _verify(manifest, staging)
            staging.rename(destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    print(f"Installed {version} into {destination}")
    if activate:
        manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
        active = resolve_path(config, config.consumer.active_manifest_path)
        active.parent.mkdir(parents=True, exist_ok=True)
        active.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Activated {version}. The service will serve it once the worker restarts.")
    else:
        print("Not activated. Re-run with --activate, or promote it from the admin Service tab.")
    return version


def _verify(manifest: dict, staged: Path) -> None:
    hashes = manifest.get("file_hashes", {})
    paths = manifest.get("paths", {})
    if not hashes or not paths:
        raise SystemExit("Manifest has no file_hashes/paths to verify against.")
    for key, expected in hashes.items():
        recorded = paths.get(key)
        if not recorded:
            raise SystemExit(f"Manifest lists a hash for {key} but no path.")
        candidate = staged / Path(str(recorded)).name
        if not candidate.exists():
            raise SystemExit(f"Bundle is missing {key} ({candidate.name}).")
        actual = sha256_file(candidate)
        if actual != expected:
            raise SystemExit(
                f"Fingerprint mismatch for {key} ({candidate.name}). "
                "The bundle is corrupt or was modified; do not use it."
            )
    print(f"Verified {len(hashes)} files against the manifest fingerprints.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export", help="Package a version as a zip.")
    exporter.add_argument("--version", default="", help="Defaults to the active version.")
    exporter.add_argument("-o", "--output", default="", help="Defaults to <version>.zip")

    importer = sub.add_parser("import", help="Verify and install a version zip.")
    importer.add_argument("archive")
    importer.add_argument("--activate", action="store_true", help="Also make it the active model.")

    args = parser.parse_args()
    config = _config(args.config)

    if args.command == "export":
        version = args.version.strip() or _active_version(config)
        output = Path(args.output) if args.output else Path(default_bundle_name(version))
        export_version(config, version, output)
        return
    import_bundle(config, Path(args.archive), activate=args.activate)


if __name__ == "__main__":
    main()
