"""The prompts of Tables 6 and 7 and the real image of each prompt.

* ``cycleprefdb``: the 380 test prompts of CyclePrefDB-T2I (Bahng et al., 2025). They summarize the dense captions of
  DCI photographs. CyclePrefDB-T2I holds only generated images. The photograph of each prompt is the image in the same
  row of the CyclePrefDB-I2T test split, which holds the same photographs in the same order.
* ``coco_val2014``: the caption with the lowest annotation id for each of the 40,504 images of MS COCO 2014 val,
  ordered by image id.

Each dataset is written once to ``<root>/data/text_to_image/<dataset>/prompts.jsonl``, one ``{"index", "prompt",
"image"}`` per line.

    pixi run python -m modalities.text_to_image.prompts
"""

import json
import tarfile
from pathlib import Path

from modalities.image_captions.datasets import COCO_URL, download_and_extract, raw_root
from unpaired_rosetta.embeddings import storage_root

NUM_PROMPTS = {"cycleprefdb": 380, "coco_val2014": 40504}


def prompts_path(
    dataset: str,
) -> Path:
    return storage_root() / "data" / "text_to_image" / dataset / "prompts.jsonl"


def load_prompts(
    dataset: str,
) -> list[dict]:
    path = prompts_path(dataset)
    if not path.exists():
        write_prompts(dataset)
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_prompts(
    dataset: str,
) -> None:
    prompts = cycleprefdb_prompts() if dataset == "cycleprefdb" else coco_prompts()
    if len(prompts) != NUM_PROMPTS[dataset]:
        raise ValueError(f"{dataset} has {len(prompts)} prompts instead of {NUM_PROMPTS[dataset]}")
    missing = [prompt["image"] for prompt in prompts if not Path(prompt["image"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images of {dataset} are missing, e.g. {missing[0]}")
    path = prompts_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(prompt) + "\n" for prompt in prompts))


def cycleprefdb_prompts() -> list[dict]:
    from huggingface_hub import hf_hub_download

    root = storage_root() / "raw" / "cycleprefdb"
    files = {"CyclePrefDB-T2I": ["test.jsonl"], "CyclePrefDB-I2T": ["test.jsonl", "test.tar.gz"]}
    for repository, names in files.items():
        for name in names:
            if not (root / repository / name).exists():
                hf_hub_download(f"carolineec/{repository}", name, repo_type="dataset", local_dir=root / repository)
    photos = root / "CyclePrefDB-I2T" / "test"
    if not photos.exists():
        with tarfile.open(root / "CyclePrefDB-I2T" / "test.tar.gz") as archive:
            archive.extractall(root / "CyclePrefDB-I2T", filter="data")
    prompts = [json.loads(line) for line in (root / "CyclePrefDB-T2I" / "test.jsonl").read_text().splitlines()]
    photographs = [json.loads(line) for line in (root / "CyclePrefDB-I2T" / "test.jsonl").read_text().splitlines()]
    return [
        {"index": index, "prompt": prompt["prompt"].strip(), "image": str(photos / photograph["image"])}
        for index, (prompt, photograph) in enumerate(zip(prompts, photographs, strict=True))
    ]


def coco_prompts() -> list[dict]:
    root = raw_root() / "coco2014"
    annotations_path = root / "annotations" / "captions_val2014.json"
    if not annotations_path.exists():
        download_and_extract(
            f"{COCO_URL}/annotations/annotations_trainval2014.zip",
            root,
            keep=lambda name: name.startswith("annotations/captions_"),
        )
    if not (root / "val2014").exists():
        download_and_extract(f"{COCO_URL}/zips/val2014.zip", root)
    annotations = json.loads(annotations_path.read_text())
    files = {image["id"]: image["file_name"] for image in annotations["images"]}
    first = {}
    for annotation in sorted(annotations["annotations"], key=lambda annotation: annotation["id"]):
        first.setdefault(annotation["image_id"], annotation)
    return [
        {"index": index, "prompt": first[image_id]["caption"].strip(), "image": str(root / "val2014" / files[image_id])}
        for index, image_id in enumerate(sorted(files))
    ]


if __name__ == "__main__":
    for name in NUM_PROMPTS:
        write_prompts(name)
        print(f"wrote {prompts_path(name)}")
