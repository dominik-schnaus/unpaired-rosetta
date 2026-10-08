"""Vision and language: image-caption corpora, zero-shot classification datasets and the encoders of the paper.

The names below are the file names of the stored embeddings (``embeddings/<dataset>/<modality>/<model>.pt``).
"""

from modalities.common import EmbeddingFile, RawData

# Datasets and models of Sec. 4.1.
TRAINING_DATASETS = {
    "coco_train2014": "MS COCO",
    "StanfordParagraphCaptioning": "SPC",
    "DenselyCaptionedImages": "DCI",
    "DOCCIDataset": "DOCCI",
}
VALIDATION_DATASET = "coco_val2014"
CLASSIFICATION_DATASETS = {"CIFAR-10": "top1", "CIFAR-100": "top5", "ImageNet-100": "top5"}  # reported accuracy

VISION_MODELS = {
    "dinov3_vit-7b16@512_mean": "DINOv3 7B/16",
    "franca_vit-g14@224_laion": "Franca G/14 (CLS)",
    "franca_vit-g14@224_laion_mean": "Franca G/14 (mean)",
    "dinov2_vit-g14@224_mean": "DINOv2 G/14",
    "dinov2_vit-b14@224_mean": "DINOv2 B/14",
    "ibot_swint_14@224_mean": "iBOT Swin-T",
    "ibot_vit-b16@224_mean": "iBOT B/16",
}
LANGUAGE_MODELS = {
    "mpnet": "MPNet",
    "qwen3-embedding-8b": "Qwen3 emb",
    "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769": "Qwen3 gen",
}
QWEN3_GEN = "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769"  # Qwen3-8B, generative token pooling, 3 repeats
# A generative model's suffix is the md5 of its prompt list. The prompt "Imagine what it would look like to see: {}"
# is used three times for Qwen3-8B and once for Qwen3-1.7B. CIFAR's language folder name holds the md5 of its 18
# templates.
QWEN3_1_7B_GEN = "qwen3-1.7b-gen_1aa1557c74992c52c60f81e5a1bd6621"

# Shapes and precision of the stored files behind the paper's numbers.
NUM_ITEMS = {
    "coco_train2014": 82783,
    "coco_val2014": 40504,
    "StanfordParagraphCaptioning": 19561,
    "DenselyCaptionedImages": 7805,
    "DOCCIDataset": 14847,
    "CIFAR-10": 50000,
    "CIFAR-100": 50000,
    "ImageNet-100": 5000,
}
NUM_CLASSES = {"CIFAR-10": 10, "CIFAR-100": 100, "ImageNet-100": 100}
CLASS_PROMPT_FOLDERS = {  # CIFAR uses 18 CLIP templates per class. ImageNet-100 uses "<name> - <WordNet definition>".
    "CIFAR-10": "language_b953684b7ed3a62cbea9fe4e30935ee2",
    "CIFAR-100": "language_b953684b7ed3a62cbea9fe4e30935ee2",
    "ImageNet-100": "language",
}
DIMENSIONS = {
    "dinov3_vit-7b16@512_mean": 4096,
    "franca_vit-g14@224_laion": 1536,
    "franca_vit-g14@224_laion_mean": 1536,
    "dinov2_vit-g14@224_mean": 1536,
    "dinov2_vit-b14@224_mean": 768,
    "ibot_swint_14@224_mean": 768,
    "ibot_vit-b16@224_mean": 768,
    "rae_dinov2_base_mean": 768,
    "mpnet": 768,
    "contriever": 768,
    "qwen3-embedding-8b": 4096,
    QWEN3_GEN: 4096,
    "qwen3-8b-mean": 4096,
    "qwen3-embedding-0.6b": 1024,
    QWEN3_1_7B_GEN: 2048,
    "qwen3-1.7b-mean": 2048,
}
# Texts stored per item. COCO has up to 7 captions (NaN-padded) and the other corpora have one. Generative pooling
# repeats its prompt three times. CIFAR has 18 templates.
TEXTS_PER_ITEM = {
    "coco_train2014": 7,
    "coco_val2014": 7,
    "CIFAR-10": 18,
    "CIFAR-100": 18,
    "coco_train2014_tokenpooling": 6,
}  # most captions among the first 5120 images
GENERATION_REPEATS = {QWEN3_GEN: 3}
# Files stored in float32 instead of bfloat16. Loading keeps their precision.
FLOAT32_FILES = {
    ("coco_train2014", "vision", "franca_vit-g14@224_laion"),
    ("coco_train2014", "vision", "franca_vit-g14@224_laion_mean"),
    ("coco_train2014", "vision", "rae_dinov2_base_mean"),
    ("coco_train2014", "language", "contriever"),
    ("coco_val2014", "vision", "franca_vit-g14@224_laion"),
    ("coco_val2014", "vision", "franca_vit-g14@224_laion_mean"),
    ("coco_val2014", "language", QWEN3_GEN),
    ("StanfordParagraphCaptioning", "vision", "franca_vit-g14@224_laion"),
    ("StanfordParagraphCaptioning", "vision", "franca_vit-g14@224_laion_mean"),
    ("StanfordParagraphCaptioning", "language", QWEN3_GEN),
    ("DenselyCaptionedImages", "vision", "franca_vit-g14@224_laion"),
    ("DenselyCaptionedImages", "vision", "franca_vit-g14@224_laion_mean"),
    ("DenselyCaptionedImages", "language", QWEN3_GEN),
    ("DOCCIDataset", "vision", "franca_vit-g14@224_laion_mean"),
    ("CIFAR-10", "vision", "franca_vit-g14@224_laion"),
    ("CIFAR-10", "vision", "franca_vit-g14@224_laion_mean"),
    ("CIFAR-100", "vision", "franca_vit-g14@224_laion"),
    ("CIFAR-100", "vision", "franca_vit-g14@224_laion_mean"),
    ("ImageNet-100", "vision", "franca_vit-g14@224_laion"),
    ("ImageNet-100", "vision", "franca_vit-g14@224_laion_mean"),
}


def vision_file(
    dataset: str,
    model: str,
) -> EmbeddingFile:
    return embedding_file(dataset, "vision", model, (NUM_ITEMS[dataset], DIMENSIONS[model]))


def language_file(
    dataset: str,
    model: str,
) -> EmbeddingFile:
    if dataset in CLASS_PROMPT_FOLDERS:
        texts = TEXTS_PER_ITEM.get(dataset, 1) * GENERATION_REPEATS.get(model, 1)
        return embedding_file(
            dataset, CLASS_PROMPT_FOLDERS[dataset], model, (NUM_CLASSES[dataset], texts, DIMENSIONS[model])
        )
    texts = TEXTS_PER_ITEM.get(dataset, 1) * GENERATION_REPEATS.get(model, 1)
    return embedding_file(dataset, "language", model, (NUM_ITEMS[dataset], texts, DIMENSIONS[model]))


def embedding_file(
    dataset: str,
    modality: str,
    model: str,
    shape: tuple[int, ...],
) -> EmbeddingFile:
    dtype = "float32" if (dataset, modality, model) in FLOAT32_FILES else "bfloat16"
    return EmbeddingFile(dataset, modality, model, shape, dtype)


def label_file(
    dataset: str,
) -> str:
    return f"embeddings/{dataset}/labels.pt"


def files_of_setting(
    dataset_x: str,
    dataset_y: str,
    vision_model: str,
    language_model: str,
    pairs_dataset: str | None,
    classification: list[str],
) -> list[EmbeddingFile]:
    """All embedding files that one vision-language experiment reads."""
    files = [vision_file(dataset_x, vision_model), language_file(dataset_y, language_model)]
    if pairs_dataset is not None:
        files += [vision_file(pairs_dataset, vision_model), language_file(pairs_dataset, language_model)]
    files += [vision_file(VALIDATION_DATASET, vision_model), language_file(VALIDATION_DATASET, language_model)]
    for dataset in classification:
        files += [vision_file(dataset, vision_model), language_file(dataset, language_model)]
    return list(dict.fromkeys(files))


ALL_VISION_MODELS = [*VISION_MODELS, "rae_dinov2_base_mean"]
LABEL_FILES = [label_file(dataset) for dataset in CLASSIFICATION_DATASETS]


def grid_files() -> list[EmbeddingFile]:
    """All files of the Sec. 4 grid: each training, validation and classification dataset with each model."""
    return [
        file
        for dataset in [*TRAINING_DATASETS, VALIDATION_DATASET, *CLASSIFICATION_DATASETS]
        for file in [
            *(vision_file(dataset, model) for model in VISION_MODELS),
            *(language_file(dataset, model) for model in LANGUAGE_MODELS),
        ]
    ]


# Text-to-image (Sec. 4.6): RAE image embeddings and two caption encoders on MS COCO train.
TEXT_TO_IMAGE_FILES = [
    vision_file("coco_train2014", "rae_dinov2_base_mean"),
    language_file("coco_train2014", "mpnet"),
    language_file("coco_train2014", "contriever"),
]

# Generative token pooling (App. D, Figs. 20 and 21): first 5 x 1024 pairs of six corpora.
TOKEN_POOLING_ITEMS = 5 * 1024  # five disjoint slices of 1024 pairs from the start of each corpus
TOKEN_POOLING_CORPORA = {
    "coco_train2014": "MS COCO",
    "cc12m": "CC12M",
    "wit_train": "WIT",
    "StanfordParagraphCaptioning": "SPC",
    "DOCCIDataset": "DOCCI",
    "DenselyCaptionedImages": "DCI",
}
TOKEN_POOLING_VISION = "dinov2_vit-b14@224_mean"
TOKEN_POOLING_LANGUAGE = {"generative": QWEN3_GEN, "embedding": "qwen3-embedding-8b", "mean": "qwen3-8b-mean"}

# Caption ladder: the same DOCCI and DCI images with less and less detailed captions, embedded by smaller models.
LADDER_LANGUAGE = {"generative": QWEN3_1_7B_GEN, "embedding": "qwen3-embedding-0.6b", "mean": "qwen3-1.7b-mean"}
LADDER_DATASETS = {  # folder -> (corpus, rung). A DCI rung names the caption type it starts from.
    **{
        f"DOCCIDataset_ladder-{level}": ("DOCCIDataset", level)
        for level in ["full", "half", "first_sentence", "keywords_12", "keywords_6", "keywords_3", "keywords_6_ordered"]
    },
    **{
        f"DenselyCaptionedImages_ladder-{rung}": ("DenselyCaptionedImages", rung)
        for rung in ["full", "extended", "short", "full-keywords_12", "full-keywords_6", "full-keywords_3"]
    },
}

NUM_ITEMS |= {f"{corpus}_tokenpooling": TOKEN_POOLING_ITEMS for corpus in TOKEN_POOLING_CORPORA}
NUM_ITEMS |= {dataset: TOKEN_POOLING_ITEMS for dataset in LADDER_DATASETS}
# The web corpora have no file of their own. Their DINOv2 embeddings are the first rows of the embeddings of the
# whole corpus (5.8M and 8.7M images), which the Hub stores as a slice.
SLICED_FROM = {
    (
        "wit_train_tokenpooling",
        "vision",
        TOKEN_POOLING_VISION,
    ): f"embeddings/wit_train/vision/{TOKEN_POOLING_VISION}.pt",
    ("cc12m_tokenpooling", "vision", TOKEN_POOLING_VISION): f"embeddings/cc12m/vision/{TOKEN_POOLING_VISION}.pt",
}


def token_pooling_file(
    corpus: str,
    model: str,
) -> EmbeddingFile:
    """File whose first ``TOKEN_POOLING_ITEMS`` rows are the pairs of ``corpus``. Uses the whole-corpus file if the
    paper has one, otherwise the subset ``<corpus>_tokenpooling``."""
    if corpus in TRAINING_DATASETS and model in (TOKEN_POOLING_VISION, QWEN3_GEN, "qwen3-embedding-8b"):
        dataset = corpus
    else:
        dataset = f"{corpus}_tokenpooling"
    return vision_file(dataset, model) if model == TOKEN_POOLING_VISION else language_file(dataset, model)


def token_pooling_files() -> list[EmbeddingFile]:
    """All files of App. D: the six corpora and the ladder rungs. The rungs reuse the images of their corpus."""
    files = [
        token_pooling_file(corpus, model)
        for corpus in TOKEN_POOLING_CORPORA
        for model in [TOKEN_POOLING_VISION, *TOKEN_POOLING_LANGUAGE.values()]
    ]
    files += [language_file(dataset, model) for dataset in LADDER_DATASETS for model in LADDER_LANGUAGE.values()]
    return files


def paper_files() -> list[EmbeddingFile]:
    """All vision-language embedding files behind a number or figure of the paper."""
    return list(dict.fromkeys([*grid_files(), *TEXT_TO_IMAGE_FILES, *token_pooling_files()]))


# Raw data, only needed to compute embeddings instead of downloading them. Sizes of the archives and of the unpacked
# files (du --apparent-size). Only the needed files are unpacked and the archives are deleted.
SPC_IMAGES = 2_592_638_399  # 19,551 distinct paragraph images of the 108,249 Visual Genome images
DCI_PHOTOS = 7_896_524_503  # 7,805 annotated photos of the 11,186 in sa_000138.tar
WIT_ROW_GROUP = 471_658_110  # first row group (10,000 rows) of the first parquet shard
CC12M_SHARD = 295_649_280 + 3_181_870  # first img2dataset shard (tar + parquet)
COCO_ANNOTATIONS = 252_872_794  # annotations_trainval2014.zip. Only the two caption files (99.2 MB) are kept.
RAW_DATA = {
    "coco_train2014": RawData(
        "MS COCO 2014 train images + captions",
        download_bytes=13_510_573_713 + COCO_ANNOTATIONS,
        stored_bytes=13_497_353_529 + 99_203_174,
    ),
    "coco_val2014": RawData(
        "MS COCO 2014 val images + captions",
        download_bytes=6_645_013_297 + COCO_ANNOTATIONS,
        stored_bytes=6_639_022_895 + 99_203_174,
    ),
    "StanfordParagraphCaptioning": RawData(
        "Visual Genome images + paragraphs",
        download_bytes=9_731_705_982 + 5_471_658_058 + 2_108_680,
        stored_bytes=SPC_IMAGES + 8_188_009,
    ),
    "DenselyCaptionedImages": RawData(
        "DCI annotations + SA-1B tar sa_000138",
        download_bytes=870_557_829 + 11_349_313_299,
        stored_bytes=1_002_091_861 + DCI_PHOTOS,
    ),
    "DOCCIDataset": RawData(
        "DOCCI images + descriptions",
        download_bytes=7_592_938_768 + 11_000_214,
        stored_bytes=7_630_658_870 + 11_000_214,
    ),
    "CIFAR-10": RawData("CIFAR-10", download_bytes=170_498_071, stored_bytes=186_218_210),
    "CIFAR-100": RawData("CIFAR-100", download_bytes=169_001_437, stored_bytes=186_305_194),
    "ImageNet-100": RawData(
        "ImageNet-1k val tar + devkit (image-net.org login), cut to 100 classes",
        download_bytes=6_744_924_160 + 2_568_145,
        stored_bytes=692_045_508,
    ),
    "wit_train_tokenpooling": RawData(
        "first WIT pairs, streamed from the Hugging Face Hub", download_bytes=WIT_ROW_GROUP, stored_bytes=161_510_411
    ),
    "cc12m_tokenpooling": RawData(
        "first CC12M pairs (img2dataset crawl, see README)", download_bytes=CC12M_SHARD, stored_bytes=182_571_258
    ),
}
# Text-only files need the caption files of a corpus but not its images.
RAW_TEXT = {
    "coco": RawData("MS COCO captions", download_bytes=COCO_ANNOTATIONS, stored_bytes=99_203_174),
    "StanfordParagraphCaptioning": RawData("SPC paragraphs", download_bytes=2_108_680, stored_bytes=8_188_009),
    "DenselyCaptionedImages": RawData("DCI annotations", download_bytes=870_557_829, stored_bytes=1_002_091_861),
    "DOCCIDataset": RawData("DOCCI descriptions", download_bytes=11_000_214, stored_bytes=11_000_214),
}


def raw_data(
    file: EmbeddingFile,
) -> RawData | None:
    """What computing ``file`` downloads. None means nothing, because the class prompts are in the code."""
    corpus = file.dataset.split("_tokenpooling")[0].split("_ladder-")[0]
    if file.modality == "vision":
        return RAW_DATA[file.dataset if file.dataset in RAW_DATA else corpus]
    if file.dataset in CLASSIFICATION_DATASETS:
        return None
    if corpus in ("wit_train", "cc12m"):
        return RAW_DATA[f"{corpus}_tokenpooling"]  # a caption is kept only if its image decodes
    return RAW_TEXT["coco" if corpus.startswith("coco") else corpus]
