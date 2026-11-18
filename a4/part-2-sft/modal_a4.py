"""Prepare the Modal storage for nanochat A4 Part 2.

Uses the active Modal profile or the standard Modal token environment variables.
Run with the Python interpreter that has the Modal SDK installed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import modal
from modal.exception import AuthError


DEFAULT_VOLUME = "a4-nanochat-volume"
SOURCE_VOLUME = "nanochat-a3-d20-ablations"
CHECKPOINT = Path("base_checkpoints/d20-baseline-8-H100")
CHECKPOINT_FILES = (
    "model_009960.pt",
    "meta_009960.json",
    *(f"optim_009960_rank{rank}.pt" for rank in range(8)),
)


def copy_checkpoint(source, destination, environment):
    if source.object_id == destination.object_id:
        raise ValueError("Select a destination volume distinct from the A3 source.")
    app = modal.App("vocalocate-a4-checkpoint-copy")

    @app.function(
        image=modal.Image.debian_slim(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        volumes={
            "/source": source.with_mount_options(read_only=True),
            "/destination": destination,
        },
        timeout=1800,
        serialized=True,
    )
    def transfer():
        source_dir = Path("/source/a3") / CHECKPOINT
        target_dir = Path("/destination/a4") / CHECKPOINT
        for name in CHECKPOINT_FILES:
            if (source_dir / name).stat().st_size == 0:
                raise ValueError(f"Empty source checkpoint file: {name}")
        metadata = json.loads((source_dir / "meta_009960.json").read_text())
        if metadata["step"] != 9960:
            raise ValueError("The source metadata does not describe step 9960.")

        target_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name in CHECKPOINT_FILES:
            original = source_dir / name
            target = target_dir / name
            with original.open("rb") as reader:
                digest = hashlib.file_digest(reader, "sha256").hexdigest()
            with tempfile.TemporaryDirectory(dir=target_dir) as scratch:
                candidate = target
                if not target.exists():
                    candidate = Path(scratch) / name
                    shutil.copyfile(original, candidate)
                with candidate.open("rb") as check:
                    if hashlib.file_digest(check, "sha256").hexdigest() != digest:
                        raise IOError(f"Destination differs from source: {name}")
                if candidate != target:
                    candidate.rename(target)
            manifest[name] = {"bytes": original.stat().st_size, "sha256": digest}
            print(f"Verified {name} ({manifest[name]['bytes']:,} bytes)", flush=True)
        destination.commit()
        return manifest

    with modal.enable_output(), app.run(environment_name=environment):
        print(json.dumps(transfer.remote(), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "copy-checkpoint"), nargs="?", default="prepare")
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--environment", help="Modal environment. Defaults to the active profile's environment.")
    args = parser.parse_args()

    try:
        if args.action == "prepare":
            volume = modal.Volume.from_name(
                args.volume, environment_name=args.environment, create_if_missing=True
            ).hydrate()
            print(f"Ready: {args.volume} ({volume.object_id}); mount at /vol and use /vol/a4")
        else:
            source = modal.Volume.from_name(SOURCE_VOLUME, environment_name=args.environment).hydrate()
            destination = modal.Volume.from_name(args.volume, environment_name=args.environment).hydrate()
            copy_checkpoint(source, destination, args.environment)
    except AuthError:
        sys.exit("Modal authentication failed. Run `modal setup` for the intended workspace and retry.")


if __name__ == "__main__":
    main()
