"""The image-caption corpora and zero-shot classification datasets of the paper, downloaded from their original sources.

Every dataset lists its items in the row order of the stored embeddings. A corpus is a ``PairedDataset``. Item ``i``
is an image with its captions, and the language file is ``[items, captions, d]``, NaN-padded. A classification dataset
is a ``ClassificationDataset`` with images, labels and the prompts of every class. Its language file is
``[classes, prompts, d]``.

Raw data lives under ``<storage root>/raw`` (``$UNPAIRED_ROSETTA_RAW`` overrides it) and is downloaded on first use.
Only the files the embeddings need are unpacked, and the archives are deleted afterwards. Three sources need an
account or a signed link: ImageNet (image-net.org login), the SA-1B tar with the DCI photos (Meta's SA-1B download
page), and the CC12M images (a crawl of URLs, many of them dead by now). ``README.md`` explains
how to provide them.
"""

import functools
import io
import json
import os
import re
import shutil
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from unpaired_rosetta.settings import raw_root

# Same settings as for the stored embeddings: allow huge images and decode truncated JPEGs.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUBSET_ITEMS = 5 * 1024  # the token-pooling appendix uses the first five slices of 1024 pairs


@dataclass
class PairedDataset(torch.utils.data.Dataset):
    """Images and their captions. ``transform`` is applied to the RGB image."""

    image_paths: list[Path]
    texts: list[list[str]]
    transform: Callable | None = None

    def __len__(
        self,
    ) -> int:
        return len(self.image_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Image.Image | torch.Tensor, list[str]]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return (image if self.transform is None else self.transform(image)), self.texts[index]


@dataclass
class ClassificationDataset:
    """Images with labels (a torchvision dataset of ``(image, label)``) and the prompts of every class."""

    images: torch.utils.data.Dataset
    labels: list[int]
    class_texts: list[list[str]]


def download(
    url: str,
    path: Path,
) -> Path:
    """Download ``url`` to ``path`` unless it exists. A partial download never gets the final name."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as file:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as progress:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                progress.update(len(chunk))
    partial.rename(path)
    return path


def extract(
    archive: Path,
    destination: Path,
    keep: Callable[[str], bool] = lambda name: True,
    rename: Callable[[str], str] = lambda name: name,
) -> None:
    """Unpack the zip or tar members that pass ``keep`` to ``destination / rename(name)``."""
    print(f"extracting {archive.name} to {destination}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zip_file:
            members = [(info.filename, info) for info in zip_file.infolist() if not info.is_dir()]
            for name, info in tqdm(members, desc=archive.name):
                if keep(name):
                    _write(destination / rename(name), lambda info=info: zip_file.open(info))
        return
    with tarfile.open(archive) as tar_file:
        for member in tqdm(tar_file, desc=archive.name):
            if member.isfile() and keep(member.name):
                _write(destination / rename(member.name), lambda member=member: tar_file.extractfile(member))


def _write(
    path: Path,
    open_member: Callable,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_member() as source, open(path, "wb") as target:
        shutil.copyfileobj(source, target)


def download_and_extract(
    url: str,
    directory: Path,
    keep: Callable[[str], bool] = lambda name: True,
    rename: Callable[[str], str] = lambda name: name,
) -> None:
    archive = download(url, directory / url.split("?")[0].split("/")[-1])
    extract(archive, directory, keep, rename)
    archive.unlink()


# MS COCO 2014 (Chen et al., 2015): images sorted by id, captions in the order of the annotation file.

COCO_URL = "http://images.cocodataset.org"


@functools.cache
def coco_captions(
    split: str,
) -> tuple[list[dict], list[list[str]]]:
    """Images of ``split`` (``train2014`` or ``val2014``) sorted by id, and the captions of each image."""
    root = raw_root() / "coco2014"
    annotations = root / "annotations" / f"captions_{split}.json"
    if not annotations.exists():
        download_and_extract(
            f"{COCO_URL}/annotations/annotations_trainval2014.zip",
            root,
            keep=lambda name: name.startswith("annotations/captions_"),
        )
    data = json.loads(annotations.read_text())
    captions = defaultdict(list)
    for annotation in data["annotations"]:
        captions[annotation["image_id"]].append(annotation["caption"])
    images = sorted(data["images"], key=lambda image: image["id"])
    return images, [captions[image["id"]] for image in images]


def coco(
    split: str,
) -> PairedDataset:
    root = raw_root() / "coco2014"
    images, texts = coco_captions(split)
    if not (root / split).exists():
        download_and_extract(f"{COCO_URL}/zips/{split}.zip", root)
    return PairedDataset([root / split / image["file_name"] for image in images], texts)


# Stanford Paragraph Captioning (Krause et al., 2017): paragraphs of Visual Genome images, in file order.

VISUAL_GENOME_URL = "https://cs.stanford.edu/people/rak248/VG_100K_2"
PARAGRAPHS_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/paragraphs_v1.json.zip"


@functools.cache
def spc_paragraphs() -> list[dict]:
    root = raw_root() / "visual_genome"
    if not (root / "paragraphs_v1.json").exists():
        download_and_extract(PARAGRAPHS_URL, root, rename=lambda name: Path(name).name)
    return json.loads((root / "paragraphs_v1.json").read_text())


def spc() -> PairedDataset:
    """Visual Genome splits its images over ``VG_100K`` and ``VG_100K_2``. Only the paragraph images are unpacked."""
    root = raw_root() / "visual_genome"
    paragraphs = spc_paragraphs()
    if not (root / "VG_100K").exists():
        needed = {f"{entry['image_id']}.jpg" for entry in paragraphs}
        for archive, folder in [("images.zip", "VG_100K"), ("images2.zip", "VG_100K_2")]:
            download_and_extract(f"{VISUAL_GENOME_URL}/{archive}", root, keep=lambda name: Path(name).name in needed)
            (root / folder).mkdir(exist_ok=True)
    paths = [
        _first_existing(root / folder / f"{entry['image_id']}.jpg" for folder in ["VG_100K", "VG_100K_2"])
        for entry in paragraphs
    ]
    return PairedDataset(paths, [[entry["paragraph"]] for entry in paragraphs])


def _first_existing(
    paths: Iterator[Path],
) -> Path:
    paths = list(paths)
    return next((path for path in paths if path.is_file()), paths[0])


# Densely Captioned Images (Urbanek et al., 2024): SA-1B photos, one annotation file per image, sorted by name.

DCI_ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz"
DCI_CAPTION_TYPES = ["short", "extended", "full"]


@functools.cache
def dci_captions() -> tuple[list[str], dict[str, list[str]]]:
    """The photo of each annotation (sorted by file name) and its caption of each type."""
    root = raw_root() / "densely_captioned_images"
    if not (root / "complete").exists():
        download_and_extract(
            DCI_ANNOTATIONS_URL, root.parent, keep=lambda name: name.startswith("densely_captioned_images/complete/")
        )
    annotations = [json.loads(path.read_text()) for path in sorted((root / "complete").iterdir())]
    photos = [Path(annotation["image"]).name for annotation in annotations]
    return photos, {
        caption_type: [dci_caption(annotation, caption_type) for annotation in annotations]
        for caption_type in DCI_CAPTION_TYPES
    }


def dci_caption(
    annotation: dict,
    caption_type: str,
) -> str:
    """``short`` is one sentence. ``extended`` adds the description of the whole image and is the paper's corpus.
    ``full`` also adds the captions of all usable masks."""
    short = annotation.get("short_caption", "")
    if caption_type == "short":
        return short
    extra = annotation.get("extra_caption", "")
    extended = f"{short}\n{extra}".strip() if extra else short
    if caption_type == "extended":
        return extended
    masks = annotation.get("mask_data", {})
    mask_texts = [text for key in annotation.get("mask_keys", []) if (text := _mask_caption(masks.get(key, {})))]
    if not mask_texts:
        return extended
    return f"{extended}\nThe following can also be seen in the image:\n" + "\n".join(mask_texts)


def _mask_caption(
    mask: dict,
) -> str | None:
    """Text of one mask. Unusable masks (quality 2) give None and coarse ones (quality 1) give the label."""
    if mask.get("mask_quality", 2) == 2:
        return None
    if mask.get("mask_quality") == 1:
        return mask.get("label")
    label, caption = mask.get("label", ""), mask.get("caption", "")
    if label and caption:
        return f"{label}: {caption}"
    return label or caption or None


def dci() -> PairedDataset:
    """The photos come from the SA-1B tar ``sa_000138.tar``. Meta gives its link on the SA-1B download page.
    Set ``$DCI_IMAGES_URL`` to it, or put the tar into ``<raw>/densely_captioned_images``."""
    root = raw_root() / "densely_captioned_images"
    names, captions = dci_captions()
    if not (root / "photos").exists():
        archive = root / "sa_000138.tar"
        if not archive.exists():
            if "DCI_IMAGES_URL" not in os.environ:
                raise FileNotFoundError(
                    "DCI needs sa_000138.tar of SA-1B: request the links on "
                    "https://ai.meta.com/datasets/segment-anything-downloads/ and set "
                    f"$DCI_IMAGES_URL to the link of sa_000138.tar (or put the tar into {root})."
                )
            download(os.environ["DCI_IMAGES_URL"], archive)
        needed = set(names)
        extract(
            archive, root / "photos", keep=lambda name: Path(name).name in needed, rename=lambda name: Path(name).name
        )
        archive.unlink()
    return PairedDataset([root / "photos" / name for name in names], [[caption] for caption in captions["extended"]])


# DOCCI (Onoe et al., 2024): one long description per image, in the order of the description file.

DOCCI_URL = "https://storage.googleapis.com/docci/data"


@functools.cache
def docci_descriptions() -> list[dict]:
    path = download(f"{DOCCI_URL}/docci_descriptions.jsonlines", raw_root() / "docci" / "docci_descriptions.jsonlines")
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [entry for entry in entries if entry.get("image_file") and entry.get("description")]


def docci() -> PairedDataset:
    root = raw_root() / "docci"
    entries = docci_descriptions()
    if not (root / "images").exists():
        download_and_extract(f"{DOCCI_URL}/docci_images.tar.gz", root)
    return PairedDataset(
        [root / "images" / entry["image_file"] for entry in entries], [[entry["description"]] for entry in entries]
    )


# Web corpora of the token-pooling appendix: the first SUBSET_ITEMS pairs with a caption and a decodable image.

WIT_REPOSITORY = "wikimedia/wit_base"  # WIT (Srinivasan et al., 2021) with its images, on the Hugging Face Hub
CC12M_TSV_URL = "https://storage.googleapis.com/conceptual_12m/cc12m.tsv"


def web_subset(
    name: str,
) -> PairedDataset:
    """``wit_train_tokenpooling`` or ``cc12m_tokenpooling``, stored as ``images/<index>`` and ``captions.json``."""
    root = raw_root() / name
    if not (root / "captions.json").exists():
        pairs = wit_pairs() if name == "wit_train_tokenpooling" else cc12m_pairs()
        _store_pairs(root, pairs)
    captions = json.loads((root / "captions.json").read_text())
    return PairedDataset(
        [root / "images" / f"{index:05d}" for index in range(len(captions))], [[caption] for caption in captions]
    )


def _store_pairs(
    root: Path,
    pairs: Iterator[tuple[bytes, str]],
) -> None:
    """Store the first SUBSET_ITEMS pairs. Pairs without a caption or with a broken image are skipped."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    captions = []
    for image_bytes, caption in pairs:
        if caption is None:
            continue
        try:
            Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as error:  # skipped, as for the paper
            print(f"skipping an image that does not decode: {error}")
            continue
        (root / "images" / f"{len(captions):05d}").write_bytes(image_bytes)
        captions.append(caption)
        if len(captions) == SUBSET_ITEMS:
            break
    (root / "captions.json").write_text(json.dumps(captions))


def wit_pairs() -> Iterator[tuple[bytes, str]]:
    """WIT streamed from the Hub in shard order. Only the first parquet shard is read."""
    from datasets import Image as HfImage
    from datasets import load_dataset

    stream = load_dataset(WIT_REPOSITORY, split="train", streaming=True).cast_column("image", HfImage(decode=False))
    for item in stream:
        yield item["image"]["bytes"], item["caption_attribution_description"]


def cc12m_pairs() -> Iterator[tuple[bytes, str]]:
    """Samples of the img2dataset shards in ``<raw>/cc12m`` in tar order. ``README.md`` describes the crawl."""
    shards = sorted((raw_root() / "cc12m").glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"CC12M needs img2dataset shards in {raw_root() / 'cc12m'}, see README.md")
    for shard in shards:
        with tarfile.open(shard) as tar_file:
            sample, key = {}, None
            for member in tar_file:
                member_key, _, extension = member.name.partition(".")
                if member_key != key and sample:
                    yield _cc12m_pair(sample)
                    sample = {}
                key = member_key
                sample[extension] = tar_file.extractfile(member).read()
            if sample:
                yield _cc12m_pair(sample)


def _cc12m_pair(
    sample: dict,
) -> tuple[bytes, str | None]:
    return sample.get("jpg", b""), json.loads(sample["json"]).get("caption") if "json" in sample else None


# Zero-shot classification: CIFAR-10/100 (train split, 18 CLIP templates per class) and ImageNet-100 (val split).

CIFAR_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# The class names of torchvision's CIFAR-100 metadata, with their underscores.
CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]
# The 100 classes of Tian et al. (2020) in WordNet id order, each with its WordNet definition.
IMAGENET100_CLASS_LIST_URL = (
    "https://raw.githubusercontent.com/danielchyeh/ImageNet-100-Pytorch/refs/heads/main/IN100.txt"
)
IMAGENET100_PROMPTS = [
    "American robin bird - large American thrush having a rust-red breast and abdomen",
    "Gila monster lizard - large orange and black lizard of southwestern United States; not dangerous unless molested",
    "Hognose snake - harmless North American snake with upturned nose; may spread its head and neck or play dead when disturbed",
    "Garter snake - any of numerous nonvenomous longitudinally-striped viviparous North American and Central American snakes",
    "Green mamba - green phase of the black mamba",
    "Spider - a spider common in European gardens",
    "Lorikeet parrot bird - any of various small brightly colored Australasian parrots having a brush-tipped tongue for feeding on nectar and soft fruits",
    "Goose - web-footed long-necked typically gregarious migratory aquatic birds usually larger and less aquatic than ducks",
    "Rock crab - crab of eastern coast of North America",
    "Fiddler crab - burrowing crab of American coastal regions having one claw much enlarged in the male",
    "Lobster - lobster of Atlantic coast of America",
    "Little blue heron bird - small bluish-grey heron of the western hemisphere",
    "American coot bird - a coot found in North America",
    "Chihuahua dog - an old breed of tiny short-haired dog with protruding eyes from Mexico held to antedate Aztec civilization",
    "Shih-Tzu dog - a Chinese breed of small dog similar to a Pekingese",
    "Papillon dog - small slender toy spaniel with erect ears and a black-spotted brown to white coat",
    "Toy terrier dog - a small active dog",
    "Walker foxhound dog - an American breed of foxhound",
    "English foxhound dog - an English breed slightly larger than the American foxhounds originally used to hunt in packs",
    "Russian wolfhound dog - tall fast-moving dog breed",
    "Saluki gazelle hound dog - old breed of tall swift keen-eyed hunting dogs resembling greyhounds; from Egypt and southwestern Asia",
    "American Staffordshire pit bull terrier dog - American breed of muscular terriers with a short close-lying stiff coat",
    "Chesapeake Bay retriever dog - American breed having a short thick oily coat ranging from brown to light tan",
    "Hungarian pointer dog - Hungarian hunting dog resembling the Weimaraner but having a rich deep red coat",
    "Kuvasz dog - long-established Hungarian breed of tall light-footed but sturdy white dog; used also as a hunting dog",
    "Komondor dog - Hungarian breed of large powerful shaggy-coated white dog; used also as guard dog",
    "Rottweiler dog - German breed of large vigorous short-haired cattle dogs",
    "Doberman pinscher dog - medium large breed of dog of German origin with a glossy black and tan coat; used as a watchdog",
    "Boxer dog - a breed of stocky medium-sized short-haired dog with a brindled coat and square-jawed muzzle developed in Germany",
    "Great Dane dog - very large powerful smooth-coated breed of dog",
    "Standard poodle dog - a breed or medium-sized poodles",
    "Mexican hairless dog - any of an old breed of small nearly hairless dogs of Mexico",
    "Coyote - small wolf native to western North America",
    "Wild African hunting dog - a powerful doglike mammal of southern and eastern Africa that hunts in large packs; now rare in settled area",
    "Red fox - the common Old World fox; having reddish-brown fur; commonly considered a single circumpolar species",
    "Tabby cat - a cat with a grey or tawny coat mottled with black",
    "Meerkat - a mongoose-like viverrine of South Africa having a face like a lemur and only four toes",
    "Beetle - any of numerous beetles that roll balls of dung on which they feed and in which they lay eggs",
    "Walking stick insect - any of various mostly tropical insects having long twiglike bodies",
    "Leafhopper - small leaping insect that sucks the juices of plants",
    "Hare - swift timid long-eared mammal larger than a rabbit having a divided upper lip and long hind legs; young born furred and with open eyes",
    "Wild boar - Old World wild swine having a narrow body and prominent tusks from which most domestic swine come; introduced in United States",
    "Gibbon - smallest and most perfectly anthropoid arboreal ape having long arms and no tail; of southern Asia and East Indies",
    "Langur - slender long-tailed monkey of Asia",
    "Ambulance - a vehicle that takes people to and from hospitals",
    "Handrail - a railing at the side of a staircase or balcony to prevent people from falling",
    "Bassinet - a basket (usually hooded) used as a baby's bed",
    "Boathouse - a shed at the edge of a river or lake; used to store boats",
    "Poke bonnet - a hat tied under the chin",
    "Bottlecap - a cap that seals a bottle",
    "Car wheel - a wheel that has a tire and rim and hubcap; used to propel the car",
    "Gong - a percussion instrument consisting of a set of tuned bells that are struck with a hammer; used as an orchestral instrument",
    "Movie theater - a theater where films are shown",
    "Cocktail shaker - a shaker for mixing cocktails",
    "Computer keyboard - a keyboard that is a data input device for computers; arrangement of keys is modelled after the typewriter keyboard",
    "Dutch oven - an oven consisting of a metal box for cooking in front of a fire",
    "Football helmet - a padded helmet with a face mask to protect the head of football players",
    "Gasmask - a protective mask with a filter; protects the face and lungs against poisonous gases",
    "Hard disk - a rigid magnetic disk mounted permanently in a drive unit",
    "Harmonica - a small rectangular free-reed instrument having a row of free reeds set back in air holes and played by blowing into the desired hole",
    "Honeycomb - a framework of hexagonal cells resembling the honeycomb built by bees",
    "Smoothing iron - home appliance consisting of a flat metal base that is heated and used to smooth cloth",
    "Denim jeans - (usually plural) close-fitting trousers of heavy denim for manual work or casual wear",
    "Lamp shade - a protective ornamental shade used to screen a light bulb from direct view",
    "Laptop computer - a portable computer small enough to use in your lap",
    "Milk can - large can for transporting milk",
    "Mixing bowl - bowl used with an electric mixer",
    "Modem - (from a combination of MOdulate and DEModulate) electronic equipment consisting of a device used to connect computers by a telephone line",
    "Moped - a motorbike that can be pedaled or driven by a low-powered gasoline engine",
    "Mortarboard - an academic cap with a flat square with a tassel on top",
    "Mousetrap - a trap for catching mice",
    "Obelisk - a stone pillar having a rectangular cross section tapering towards a pyramidal top",
    "Park bench - a bench in a public park",
    "Pedestal - an architectural support or base (as for a column or statue)",
    "Pickup truck - a light truck with an open body and low sides and a tailboard",
    "Pirate ship - a ship that is manned by pirates",
    "Purse - a small bag for carrying money",
    "Fishing casting reel - winder consisting of a revolving spool with a handle; attached to a fishing rod",
    "Rocking chair - a chair mounted on rockers",
    "rotisserie - an oven or broiler equipped with a rotating spit on which meat cooks as it turns",
    "Safety pin - a pin in the form of a clasp; has a guard so the point of the pin will not stick the user",
    "Sarong - a loose skirt consisting of brightly colored fabric wrapped around the body; worn by both women and men in the South Pacific",
    "Ski mask - a woolen face mask to protect the face from cold while skiing on snow",
    "Slide rule - analog computer consisting of a handheld instrument used for rapid calculations; have been replaced by pocket calculators",
    "Stretcher - a litter for transporting people who are ill or wounded or dead; usually consists of a sheet of canvas stretched between two poles",
    "Theater curtain - a hanging cloth that conceals the stage from the view of the audience; rises or parts at the beginning and descends or closes between acts and at the end of a performance",
    "Throne - the chair of state for a monarch, bishop, etc.",
    "Tile roof - a roof made of fired clay tiles",
    "Tripod - a three-legged rack used for support",
    "Hot tub - a large open vessel for holding or storing liquids",
    "Vacuum cleaner - an electrical home appliance that cleans by suction",
    "Window screen - screen to keep insects from entering a building through the open window",
    "Airplane wing - one of the horizontal airfoils on either side of the fuselage of an airplane",
    "Cabbage - any of several varieties of cabbage having a large compact globular head; may be steamed or boiled or stir-fried or used raw in coleslaw",
    "Cauliflower - compact head of white undeveloped flowers",
    "Pineapple - large sweet fleshy tropical fruit with a terminal tuft of stiff leaves; widely cultivated",
    "Carbonara - sauce for pasta; contains eggs and bacon or ham and grated cheese",
    "Chocolate syrup - sauce made with unsweetened chocolate or cocoa and sugar and water",
    "Gyromitra mushroom - any fungus of the genus Gyromitra",
    "Stinkhorn mushroom - any of various ill-smelling brown-capped fungi of the order Phallales",
]


def class_texts(
    name: str,
) -> list[list[str]]:
    """The prompts of each class, in label order."""
    if name == "ImageNet-100":
        return [[prompt] for prompt in IMAGENET100_PROMPTS]
    classes = CIFAR10_CLASSES if name == "CIFAR-10" else CIFAR100_CLASSES
    return [[template.format(name) for template in CIFAR_TEMPLATES] for name in classes]


def classification_dataset(
    name: str,
) -> ClassificationDataset:
    if name == "ImageNet-100":
        from torchvision.datasets import ImageFolder

        images = ImageFolder(imagenet100_root() / "val")
        return ClassificationDataset(images, images.targets, class_texts(name))
    from torchvision.datasets import CIFAR10, CIFAR100

    root = raw_root() / "cifar"
    cifar = CIFAR10 if name == "CIFAR-10" else CIFAR100
    missing = not (root / cifar.base_folder).exists()
    images = cifar(root, train=True, download=missing)
    if missing:
        (root / cifar.filename).unlink()
    return ClassificationDataset(images, images.targets, class_texts(name))


def imagenet100_root() -> Path:
    """ImageNet-100 is cut from the ImageNet-1k validation set, which needs an image-net.org account.
    Put ``ILSVRC2012_img_val.tar`` and ``ILSVRC2012_devkit_t12.tar.gz`` into ``<raw>/imagenet``."""
    root = raw_root() / "imagenet100"
    if (root / "val").exists():
        return root
    from torchvision.datasets.imagenet import parse_devkit_archive

    imagenet = raw_root() / "imagenet"
    parse_devkit_archive(imagenet)  # WordNet id of each validation image, in file name order
    _, validation_wnids = torch.load(imagenet / "meta.bin", weights_only=False)
    classes = set(download(IMAGENET100_CLASS_LIST_URL, root / "IN100.txt").read_text().split())
    with tarfile.open(imagenet / "ILSVRC2012_img_val.tar") as tar_file:
        members = sorted((member for member in tar_file if member.isfile()), key=lambda member: member.name)
        for member, wnid in zip(tqdm(members, desc="ImageNet-100"), validation_wnids, strict=True):
            if wnid in classes:
                _write(root / "val" / wnid / member.name, lambda member=member: tar_file.extractfile(member))
    return root


# Caption ladder of the token-pooling appendix: captions of the first SUBSET_ITEMS images, shortened by fixed rules.

# Function words carry no visual content. Dropping them turns a description into a tag list.
STOPWORDS = frozenset(
    """
a an the this that these those there here it its it's they them their theirs he him his she her hers we us our
ours you your yours i me my mine who whom whose which what where when why how
is are was were be been being am do does did done doing have has had having can could shall should will would may
might must
and or but nor so yet if then than as because while although though however also too very quite rather just only
even still both each either neither all any some no not none more most much many few less least other another same
of in on at to from by with without within into onto out over under above below up down off across through during
before after between among against about around near behind beside beyond along toward towards upon per via
s t re ve ll d m o
""".split()
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def content_words(
    caption: str,
) -> list[str]:
    """Words without function words and duplicates, in the order they appear."""
    seen, kept = set(), []
    for word in WORD.findall(caption):
        if word.lower() not in STOPWORDS and word.lower() not in seen:
            seen.add(word.lower())
            kept.append(word)
    return kept


def keywords(
    caption: str,
    count: int,
    frequencies: Counter | None,
) -> str:
    """``count`` comma-separated content words in the order they appear. Picks the rarest ones in the corpus, or the
    first ones if ``frequencies`` is None."""
    words = content_words(caption)
    if frequencies is None:
        return ", ".join(words[:count])
    rarest = sorted(words, key=lambda word: (frequencies.get(word.lower(), 0), word.lower()))[:count]
    return ", ".join(sorted(rarest, key=words.index))


def first_sentences(
    caption: str,
    count: int | None = None,
    fraction: float | None = None,
) -> str:
    """The first ``count`` sentences, or the first ``fraction`` of them. Keeps at least one."""
    sentences = [sentence for sentence in SENTENCE_END.split(caption.strip()) if sentence]
    keep = count if count is not None else round(len(sentences) * fraction)
    return " ".join(sentences[: max(1, keep)])


LADDER_LEVELS = {
    "full": lambda caption, frequencies: caption,
    "half": lambda caption, frequencies: first_sentences(caption, fraction=0.5),
    "first_sentence": lambda caption, frequencies: first_sentences(caption, count=1),
    "keywords_12": lambda caption, frequencies: keywords(caption, 12, frequencies),
    "keywords_6": lambda caption, frequencies: keywords(caption, 6, frequencies),
    "keywords_3": lambda caption, frequencies: keywords(caption, 3, frequencies),
    "keywords_6_ordered": lambda caption, frequencies: keywords(caption, 6, None),  # control that keeps the first words
}


def ladder_texts(
    corpus: str,
    caption_type: str | None,
    level: str,
) -> list[list[str]]:
    """Captions of the first SUBSET_ITEMS images of DOCCI or DCI (in ``caption_type``) at one detail level."""
    if corpus == "DOCCIDataset":
        captions = [entry["description"] for entry in docci_descriptions()[:SUBSET_ITEMS]]
    else:
        captions = dci_captions()[1][caption_type][:SUBSET_ITEMS]
    frequencies = Counter()  # number of captions that contain each word
    for caption in captions:
        frequencies.update({word.lower() for word in content_words(caption)})
    return [[LADDER_LEVELS[level](caption, frequencies)] for caption in captions]


CORPORA = {
    "coco_train2014": lambda: coco("train2014"),
    "coco_val2014": lambda: coco("val2014"),
    "StanfordParagraphCaptioning": spc,
    "DenselyCaptionedImages": dci,
    "DOCCIDataset": docci,
    "wit_train_tokenpooling": lambda: web_subset("wit_train_tokenpooling"),
    "cc12m_tokenpooling": lambda: web_subset("cc12m_tokenpooling"),
}
CORPUS_TEXTS = {  # captions only, without downloading the images
    "coco_train2014": lambda: coco_captions("train2014")[1],
    "coco_val2014": lambda: coco_captions("val2014")[1],
    "StanfordParagraphCaptioning": lambda: [[entry["paragraph"]] for entry in spc_paragraphs()],
    "DenselyCaptionedImages": lambda: [[caption] for caption in dci_captions()[1]["extended"]],
    "DOCCIDataset": lambda: [[entry["description"]] for entry in docci_descriptions()],
}
CLASSIFICATION = ["CIFAR-10", "CIFAR-100", "ImageNet-100"]
LADDER = re.compile(r"(?P<corpus>DOCCIDataset|DenselyCaptionedImages)_ladder-(?P<suffix>.+)")


def paired_dataset(
    name: str,
) -> PairedDataset:
    """Images and captions of a corpus. ``<corpus>_tokenpooling`` is its first SUBSET_ITEMS items."""
    if name.endswith("_tokenpooling") and name not in CORPORA:
        dataset = CORPORA[name.removesuffix("_tokenpooling")]()
        return PairedDataset(dataset.image_paths[:SUBSET_ITEMS], dataset.texts[:SUBSET_ITEMS])
    return CORPORA[name]()


def texts(
    name: str,
) -> list[list[str]]:
    """Texts of each row of the language file of ``name``: captions, class prompts or ladder captions."""
    if name in CLASSIFICATION:
        return class_texts(name)
    if match := LADDER.fullmatch(name):
        return ladder_texts(*ladder_rung(match["corpus"], match["suffix"]))
    if name.endswith("_tokenpooling") and name.removesuffix("_tokenpooling") in CORPUS_TEXTS:
        return CORPUS_TEXTS[name.removesuffix("_tokenpooling")]()[:SUBSET_ITEMS]
    if name in CORPUS_TEXTS:
        return CORPUS_TEXTS[name]()
    return paired_dataset(name).texts


def ladder_rung(
    corpus: str,
    suffix: str,
) -> tuple[str, str | None, str]:
    """``(corpus, caption type, level)`` of a folder name such as ``DenselyCaptionedImages_ladder-full-keywords_6``."""
    if corpus == "DOCCIDataset":
        return corpus, None, suffix
    caption_type, _, level = suffix.partition("-")
    return corpus, caption_type, level or "full"


def labels(
    name: str,
) -> torch.Tensor:
    return torch.tensor(classification_dataset(name).labels, dtype=torch.long)
