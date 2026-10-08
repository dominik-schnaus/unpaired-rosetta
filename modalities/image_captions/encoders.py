"""The vision and language encoders of the paper, with preprocessing and pooling to one vector per input.

Vision encoders average all tokens of their last layer (class, register and patch tokens). Franca (CLS) keeps only
the class token. Language encoders are sentence-embedding models with their own pooling, or Qwen3 with the mean over
the hidden states of the chat-formatted text. Generative token pooling needs vLLM and is in ``generative.py``.

Each encoder runs in the precision of the stored file: bfloat16 weights under bfloat16 autocast, or float32 throughout
for files stored in float32. The torch.hub repositories are pinned to fixed commits. For iBOT, Franca and DINOv3 these
are the commits the embeddings were computed with. For DINOv2 it is a later commit whose attention kernel differs in
the last bits.
"""

import os
import shutil
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn.functional import interpolate, normalize
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor, v2
from torchvision.transforms.functional import to_tensor

from unpaired_rosetta.embeddings import storage_root

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DINOV2_HUB = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV3_REPOSITORY, DINOV3_COMMIT = "facebookresearch/dinov3", "91ffb6803c74aeb6927209b82560142743763dd7"
IBOT_HUB = "NeovisionSAS/ibot:223e3c109d37ff71c5d94a73442e8cfbe96261ca"
FRANCA_HUB = "valeoai/Franca:52653cdd2f94fc7e4dd12655cf326b181a48091d"
DINOV3_WEIGHTS = "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
RAE_REPOSITORY = "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08"


def hub_model(
    repository: str,
    entry: str,
    **kwargs: object,
) -> nn.Module:
    """Load a model from a torch.hub repository pinned to a commit.

    Validation is skipped because a commit is not a branch or tag.
    """
    return torch.hub.load(repository, entry, trust_repo=True, skip_validation=True, **kwargs)


@dataclass
class VisionEncoder:
    module: nn.Module  # images [B, 3, H, W] -> embeddings [B, d]
    transform: Callable  # PIL image -> tensor [3, H, W]


class TokenMean(nn.Module):
    """Mean over the class, register and patch tokens of a DINOv2-style backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        register_key: str,
    ) -> None:
        super().__init__()
        self.backbone, self.register_key = backbone, register_key

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        features = self.backbone.forward_features(images)
        tokens = [features["x_norm_clstoken"][:, None], features[self.register_key], features["x_norm_patchtokens"]]
        return torch.cat(tokens, dim=1).mean(dim=1)


class AllTokensMean(nn.Module):
    """Mean over all last-layer tokens of iBOT."""

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        return self.backbone(images, return_all_tokens=True).mean(dim=1)


def dinov2_transform() -> Callable:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RGB(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dinov3_transform() -> Callable:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((512, 512), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def bicubic_transform() -> Callable:
    """Bicubic resize to 224 x 224, used for iBOT and Franca."""
    return Compose(
        [
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def dinov3_vit7b() -> nn.Module:
    """DINOv3 ViT-7B/16. The weights are gated. Request them on
    https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and put
    ``dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`` into ``<storage root>/weights``. You can instead set
    ``$DINOV3_WEIGHTS`` to its path or ``$DINOV3_WEIGHTS_URL`` to the download link from the e-mail.
    ``$DINOV3_REPO`` can point to a local clone instead of the pinned GitHub commit."""
    path = Path(os.environ.get("DINOV3_WEIGHTS", storage_root() / "weights" / DINOV3_WEIGHTS))
    if not path.exists():
        if "DINOV3_WEIGHTS_URL" not in os.environ:
            raise FileNotFoundError(f"DINOv3 weights not found at {path}; see the docstring of encoders.dinov3_vit7b")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(os.environ["DINOV3_WEIGHTS_URL"], str(path))
    # Import only the backbone. The repository's hubconf also imports the evaluation heads and their dependencies.
    sys.path.insert(0, str(os.environ.get("DINOV3_REPO") or github_checkout(DINOV3_REPOSITORY, DINOV3_COMMIT)))
    from dinov3.hub.backbones import dinov3_vit7b16

    with torch.device("meta"):  # 7B parameters: build without memory, then assign the checkpoint tensors
        backbone = dinov3_vit7b16(pretrained=False)
    backbone.load_state_dict(torch.load(path, map_location="cpu", mmap=True), assign=True)
    return TokenMean(backbone, "x_storage_tokens")


def github_checkout(
    repository: str,
    commit: str,
) -> Path:
    """Source of ``repository`` at ``commit``, unpacked into the torch.hub directory."""
    directory = Path(torch.hub.get_dir()) / f"{repository.replace('/', '_')}_{commit}"
    if not directory.exists():
        archive = directory.with_suffix(".zip")
        directory.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(f"https://github.com/{repository}/archive/{commit}.zip", str(archive))
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(directory.parent / f"{directory.name}.partial")
        (directory.parent / f"{directory.name}.partial" / f"{repository.split('/')[1]}-{commit}").rename(directory)
        shutil.rmtree(directory.parent / f"{directory.name}.partial")
        archive.unlink()
    return directory


def adm_center_crop(
    image: Image.Image,
    size: int,
) -> Image.Image:
    """Center crop of ADM (Dhariwal & Nichol, 2021), which RAE trains with. Halves the image with a box filter while it
    is at least twice the target size, resizes the short side to ``size`` bicubically and crops the center."""
    while min(*image.size) >= 2 * size:
        image = image.resize(tuple(side // 2 for side in image.size), resample=Image.BOX)
    scale = size / min(*image.size)
    image = image.resize(tuple(round(side * scale) for side in image.size), resample=Image.BICUBIC)
    array = np.array(image.convert("RGB"))
    top, left = (array.shape[0] - size) // 2, (array.shape[1] - size) // 2
    return Image.fromarray(array[top : top + size, left : left + size])


class RAE(nn.Module):
    """DINOv2-B encoder of the representation autoencoder (Zheng et al., 2025) used by our text-to-image model. Patch
    tokens are normalized with the checkpoint's latent statistics, averaged and scaled to unit length."""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        from diffusers import AutoModel

        self.autoencoder = AutoModel.from_pretrained(RAE_REPOSITORY, reshape_to_2d=False)

    @staticmethod
    def transform(
        image: Image.Image,
    ) -> Tensor:
        """ADM crop to 256, then the encoder's own preprocessing: bicubic resize to 224 and ImageNet normalization."""
        pixels = to_tensor(adm_center_crop(image, 256))[None]
        pixels = interpolate(pixels, size=(224, 224), mode="bicubic", align_corners=False)
        mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((pixels - mean) / std)[0]

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        tokens = self.autoencoder._encoder_forward_fn(self.autoencoder.encoder, images)  # patch tokens only
        batch, num_tokens, channels = tokens.shape
        side = int(num_tokens**0.5)
        latents = self.autoencoder._normalize_latents(tokens.transpose(1, 2).view(batch, channels, side, side))
        return normalize(latents.flatten(2).transpose(1, 2).mean(dim=1), dim=-1)  # stored rows have unit length


def vision_encoder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> VisionEncoder:
    if name == "rae_dinov2_base_mean":
        module = RAE()
        encoder = VisionEncoder(module, module.transform)
    else:
        encoder = VISION_ENCODERS[name]()
    encoder.module.to(device=device, dtype=dtype).eval()
    return encoder


VISION_ENCODERS = {
    "dinov2_vit-b14@224_mean": lambda: VisionEncoder(
        TokenMean(hub_model(DINOV2_HUB, "dinov2_vitb14"), "x_norm_regtokens"), dinov2_transform()
    ),
    "dinov2_vit-g14@224_mean": lambda: VisionEncoder(
        TokenMean(hub_model(DINOV2_HUB, "dinov2_vitg14"), "x_norm_regtokens"), dinov2_transform()
    ),
    "dinov3_vit-7b16@512_mean": lambda: VisionEncoder(dinov3_vit7b(), dinov3_transform()),
    "ibot_vit-b16@224_mean": lambda: VisionEncoder(AllTokensMean(hub_model(IBOT_HUB, "vitb_16")), bicubic_transform()),
    "ibot_swint_14@224_mean": lambda: VisionEncoder(
        AllTokensMean(hub_model(IBOT_HUB, "swint_14")), bicubic_transform()
    ),
    "franca_vit-g14@224_laion": lambda: VisionEncoder(
        hub_model(FRANCA_HUB, "franca_vitg14", weights="LAION", img_size=224, use_rasa_head=False), bicubic_transform()
    ),
    "franca_vit-g14@224_laion_mean": lambda: VisionEncoder(
        TokenMean(
            hub_model(FRANCA_HUB, "franca_vitg14", weights="LAION", img_size=224, use_rasa_head=False),
            "x_norm_regtokens",
        ),
        bicubic_transform(),
    ),
}


class SentenceEncoder(nn.Module):
    """A sentence-transformers model with its tokenizer, pooling and normalization (if any)."""

    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
        **kwargs: object,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = device
        self.model = SentenceTransformer(name, device=str(device), model_kwargs={"dtype": dtype}, **kwargs)

    def forward(
        self,
        texts: list[str],
    ) -> Tensor:
        features = self.model.preprocess(texts)
        features = {key: value.to(self.device) for key, value in features.items() if isinstance(value, Tensor)}
        return self.model.forward(features)["sentence_embedding"]


class MeanPooling(nn.Module):
    """Qwen3 with the mean of the last hidden states over the tokens of the chat-formatted text."""

    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)

    def forward(
        self,
        texts: list[str],
    ) -> Tensor:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False, enable_thinking=True
            )
            for text in texts
        ]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        mask = inputs["attention_mask"]
        # Positions count only real tokens, so a left-padded text gets the same embedding as without padding.
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=mask,
            position_ids=positions,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
        )  # no logits needed
        hidden = outputs.hidden_states[-1]
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def language_encoder(
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Callable[[list[str]], Tensor]:
    return LANGUAGE_ENCODERS[name](dtype, device).eval()


LANGUAGE_ENCODERS = {
    "mpnet": lambda dtype, device: SentenceEncoder("sentence-transformers/all-mpnet-base-v2", dtype, device),
    "contriever": lambda dtype, device: SentenceEncoder("facebook/contriever", dtype, device),
    "qwen3-embedding-8b": lambda dtype, device: SentenceEncoder("Qwen/Qwen3-Embedding-8B", dtype, device),
    "qwen3-embedding-0.6b": lambda dtype, device: SentenceEncoder("Qwen/Qwen3-Embedding-0.6B", dtype, device),
    "qwen3-8b-mean": lambda dtype, device: MeanPooling("Qwen/Qwen3-8B", dtype, device),
    "qwen3-1.7b-mean": lambda dtype, device: MeanPooling("Qwen/Qwen3-1.7B", dtype, device),
}
