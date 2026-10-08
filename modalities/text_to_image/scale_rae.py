"""Scale-RAE (Tong et al., 2026), the paired reference model of Tables 6 and 7 (``scale-rae`` environment).

Scale-RAE conditions an RAE diffusion transformer (2.4B parameters) directly on a language model (Qwen2.5-1.5B). It
is trained on paired data. Sampling follows its own GenEval script and uses the helpers of its inference CLI
(submodule ``external/Scale-RAE``): "Generate an image of <prompt>" in the Qwen-2 chat template, greedy decoding of
the image tokens, guidance 1.0, float32 (its loader supports no other dtype) and 224 pixels. The diffusion head draws
from the global generator, which is seeded with the prompt's noise seed before each prompt. A prompt without
generated image tokens gets a black image, as in the GenEval script.

    pixi run -e scale-rae python -m modalities.text_to_image.scale_rae --prompts prompts.jsonl --start 0 --stop 128 --output dir
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from modalities.text_to_image import SCALE_RAE_ROOT, prompt_seed
from modalities.text_to_image.sample import image_path, save_image

MODEL = "nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B"  # Smallest released model.
DECODER = "nyu-visionx/siglip2_decoder"
PREFIX = "Generate an image of "  # Prefix used by its GenEval script.
GUIDANCE_LEVEL = 1.0
MAX_NEW_TOKENS = 512
IMAGE_SIZE = 224


def sample_prompts(
    prompts: list[dict],
    output: Path,
) -> None:
    """Write ``<output>/<index>.png`` for each prompt. Existing images are skipped."""
    sys.path[:0] = [str(SCALE_RAE_ROOT / "inference"), str(SCALE_RAE_ROOT)]
    from cli import (
        _common_gen_kwargs,
        build_decoder,
        decode_image_embeds,
        load_model,
        prepare_special_token_ids,
        tokenize_prompt,
    )
    from scale_rae.constants import IMAGE_PLACEHOLDER
    from scale_rae.conversation import conv_templates

    tokenizer, model, _, _ = load_model(MODEL)
    decoder = build_decoder(model, model_path=MODEL, decoder_repo_id=DECODER)
    start_id, end_id, eos_id = prepare_special_token_ids(tokenizer)
    generation = _common_gen_kwargs(start_id, end_id, eos_id, GUIDANCE_LEVEL, MAX_NEW_TOKENS)
    for done, prompt in enumerate(prompts, 1):
        path = image_path(output, prompt["index"])
        if path.exists():
            continue
        conversation = conv_templates["qwen_2"].copy()
        conversation.append_message(conversation.roles[0], PREFIX + prompt["prompt"].replace(IMAGE_PLACEHOLDER, ""))
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenize_prompt(conversation.get_prompt(), tokenizer, model.device)
        torch.manual_seed(prompt_seed(prompt["index"]))
        with torch.inference_mode():
            _, image_embeds = model.generate(input_ids, images=None, **generation)
        images = (
            decode_image_embeds(model, image_embeds, decoder)
            if image_embeds is not None and image_embeds.numel()
            else []
        )
        save_image(images[0] if images else Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), path)
        if done % 32 == 0:
            print(f"scale-rae: {done}/{len(prompts)} images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="prompts.jsonl of modalities.text_to_image.prompts")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prompts = [json.loads(line) for line in arguments.prompts.read_text().splitlines()][
        arguments.start : arguments.stop
    ]
    sample_prompts(prompts, arguments.output)


if __name__ == "__main__":
    main()
