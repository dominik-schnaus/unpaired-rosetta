"""Publish the stored vision-language embeddings of the paper on the Hugging Face Hub (``HF_REPOSITORY``).

    python -m modalities.image_captions.upload_to_hub --dry-run   # list the files and their total size
    python -m modalities.image_captions.upload_to_hub             # upload (needs a token with write access)

Each file keeps its path relative to the storage root (``embeddings/<dataset>/<modality>/<model>.pt``), where
``download_precomputed`` looks for it. Files are uploaded byte for byte, except the DINOv2 embeddings of the WIT and
CC12M subsets. Those are the first rows of the whole-corpus embeddings (``SLICED_FROM``), and only that slice is
uploaded.
"""

import argparse
from pathlib import Path

import torch

from modalities.common import HF_REPOSITORY, format_bytes
from modalities.image_captions import LABEL_FILES, SLICED_FROM, paper_files
from unpaired_rosetta.embeddings import storage_root


def uploads() -> list[tuple[str, Path, int | None]]:
    """``(path in the repository, stored source, rows to keep or None for the whole file)`` for each file."""
    entries = []
    for file in paper_files():
        source = SLICED_FROM.get((file.dataset, file.modality, file.model))
        rows = file.shape[0] if source is not None else None
        entries.append((file.relative_path, storage_root() / (source or file.relative_path), rows))
    entries += [(label_file, storage_root() / label_file, None) for label_file in LABEL_FILES]
    return entries


def upload_size(
    source: Path,
    rows: int | None,
) -> int:
    if rows is None:
        return source.stat().st_size
    stored = torch.load(source, mmap=True)
    return rows * stored[0].numel() * stored.element_size()


def stage(
    entries: list[tuple[str, Path, int | None]],
    folder: Path,
) -> None:
    """Build a folder with the repository layout. Whole files are symlinked and slices are saved as new files."""
    for relative, source, rows in entries:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        if rows is None:
            target.symlink_to(source.resolve())
        else:
            torch.save(torch.load(source, mmap=True)[:rows].clone(), target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only list the files and their total size")
    arguments = parser.parse_args()

    entries = uploads()
    missing = [relative for relative, source, _ in entries if not source.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} stored files are missing, e.g. {missing[:3]}")
    sizes = [upload_size(source, rows) for _, source, rows in entries]
    for (relative, _, rows), size in zip(entries, sizes):
        print(f"{format_bytes(size):>10}  {relative}{'  (first rows of the corpus file)' if rows else ''}")
    print(f"{len(entries)} files, {format_bytes(sum(sizes))} to {HF_REPOSITORY}")
    if arguments.dry_run:
        return

    from huggingface_hub import HfApi

    folder = storage_root() / "hub_upload"
    stage(entries, folder)
    api = HfApi()
    api.create_repo(HF_REPOSITORY, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(repo_id=HF_REPOSITORY, folder_path=folder, repo_type="dataset")


if __name__ == "__main__":
    main()
