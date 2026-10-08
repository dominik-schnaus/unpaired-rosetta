"""TIFA (Hu et al., 2023) as the official code computes it (github.com/Yushi-Hu/tifa, ``tifascore``). It uses the
open LLaMA-2 question generator instead of GPT-3.5 and BLIP-large as the VQA model (``t2i-metrics`` env).

1. ``question_gen_llama2``: the generator writes typed multiple-choice questions for a caption (beam search, 5 beams,
   512 tokens).
2. ``question_filter``: a question is kept only if UnifiedQA answers it correctly from the caption alone. Unless the
   answer is yes or no, the free-form answer must also reach a token F1 above 0.6.
3. ``VQAModel("blip-large").multiple_choice_vqa``: BLIP answers each kept question on the image. An answer that is
   not one of the choices counts as the closest choice under SBERT (mean-pooled all-mpnet-base-v2). The TIFA score
   of an image is the fraction of correct answers, or NaN without questions.

The code runs in batches but gives the same results as the official one-at-a-time calls. Padding changes the beams of
the generator and the answers of UnifiedQA, so each of their batches only holds prompts of one token length. BLIP's
question padding is masked in the cross-attention of its answer decoder. ``tests/tifa/test_equivalence.py`` compares
with the official code.

    pixi run -e t2i-metrics python -m modalities.text_to_image.tifa --prompts prompts.jsonl --start 0 --stop 128 --output questions.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PIL import Image
    from torch import Tensor
    from transformers import BlipForQuestionAnswering, BlipProcessor, MPNetModel, PreTrainedTokenizerBase

GENERATOR = "tifa-benchmark/llama2_tifa_question_generation"
FILTER = "allenai/unifiedqa-v2-t5-large-1363200"
VQA = "Salesforce/blip-vqa-capfilt-large"
SBERT = "sentence-transformers/all-mpnet-base-v2"
INTRODUCTION = (
    "Given an image description, generate one or two multiple-choice questions that verifies if the image description "
    "is correct.\nClassify each concept into a type (object, human, animal, food, activity, attribute, counting, color, "
    "material, spatial, location, shape, other), and then generate a question for each type.\n"
)
CATEGORIES = [
    "object",
    "human",
    "animal",
    "food",
    "activity",
    "attribute",
    "counting",
    "color",
    "material",
    "spatial",
    "location",
    "shape",
    "other",
]
FREE_FORM_THRESHOLD = 0.6


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def completions(
    captions: list[str],
    batch_size: int,
    known: dict[int, str],
    save: Callable[[dict[int, str]], None] | None = None,
) -> list[str]:
    """Generator output for each caption, cut at the first blank line (``llama2_completion``).

    ``known`` holds completions of an interrupted call by position. ``save`` gets all completions after each batch.
    """
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(GENERATOR)
    model = transformers.AutoModelForCausalLM.from_pretrained(GENERATOR, torch_dtype=torch.float16).eval().to(device())
    prompts = [
        f"<s>[INST] <<SYS>>\n{INTRODUCTION}\n<</SYS>>\n\nDescription: {caption} [/INST] Entities:"
        for caption in captions
    ]
    encoded = [tokenizer(prompt).input_ids for prompt in prompts]
    by_length = defaultdict(list)
    for index, ids in enumerate(encoded):
        if index not in known:
            by_length[len(ids)].append(index)
    for _, indices in sorted(by_length.items()):
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            input_ids = torch.tensor([encoded[i] for i in chunk], device=device())
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    do_sample=False,
                    num_beams=5,
                    num_return_sequences=1,
                    max_length=512,
                    pad_token_id=tokenizer.unk_token_id,
                )
            for i, sequence in zip(chunk, output):
                # Same as the pipeline's `generated_text[len(prompt):]`.
                text = tokenizer.decode(sequence, skip_special_tokens=True)
                known[i] = text[len(tokenizer.decode(encoded[i], skip_special_tokens=True)) :].split("\n\n")[0]
            if save is not None:
                save(known)
            print(f"TIFA questions: {len(known)}/{len(captions)} captions", flush=True)
    del model
    torch.cuda.empty_cache()
    return [known[i] for i in range(len(captions))]


def parse(
    caption: str,
    completion: str,
) -> list[dict]:
    """``parse_resp`` and ``get_llama2_question_and_answers``, line for line.

    The official parser starts at the seventh line, as the GPT-3 format needs. The LLaMA-2 generator writes its first
    ``About ...:`` header on the sixth line, so the first question of each caption has no entity and is dropped, as in
    the official code. A header without " (" crashes the official parser. Here it only clears the entity.
    """
    entity = element_type = question = choices = None
    instances = []
    for line in completion.split("\n")[6:]:
        if line.startswith("About "):
            whole = line[len("About ") : -1]
            if " (" in whole:
                entity, element_type = whole.split(" (")[0], whole.split(" (")[1].split(")")[0]
            else:
                entity = element_type = None
        elif line.startswith("Q: "):
            question = line[3:]
        elif line.startswith("Choices: "):
            choices = line[9:].split(", ")
        elif line.startswith("A: "):
            if entity and question and choices:
                instances.append((entity, question, choices, line[3:], element_type))
            question = choices = None
    return [
        {
            "caption": caption,
            "element": entity,
            "question": question,
            "choices": choices,
            "answer": answer,
            "element_type": "animal/human" if element_type in ("animal", "human") else element_type,
        }
        for entity, question, choices, answer, element_type in instances
        if element_type in CATEGORIES
    ]


def token_f1(
    gold: list[str],
    predicted: list[str],
) -> float:
    """``question_filter.compute_prf``."""
    if not gold:
        return 1.0 if not predicted else 0.0
    true_positives = sum(token in predicted for token in gold)
    false_negatives = len(gold) - true_positives
    false_positives = sum(token not in gold for token in predicted)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def filter_questions(
    generated: list[list[dict]],
    batch_size: int = 64,
) -> list[list[dict]]:
    """``question_filter.filter_question_and_answers`` with ``unifiedqa.UnifiedQAModel``.

    Keeps each first occurrence of a question that UnifiedQA answers correctly from the caption alone.
    """
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from word2number import w2n

    tokenizer = T5Tokenizer.from_pretrained(FILTER)
    model = T5ForConditionalGeneration.from_pretrained(FILTER).eval().to(device())

    def answer(
        prompts: list[str],
    ) -> list[str]:
        encoded = [tokenizer.encode(prompt) for prompt in prompts]
        by_length = defaultdict(list)
        for index, ids in enumerate(encoded):
            by_length[len(ids)].append(index)
        answers = [None] * len(prompts)
        for indices in by_length.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                with torch.inference_mode():
                    output = model.generate(
                        torch.tensor([encoded[i] for i in chunk], device=device()), max_new_tokens=30
                    )
                for i, text in zip(chunk, tokenizer.batch_decode(output, skip_special_tokens=True)):
                    answers[i] = text
        return answers

    candidates = []  # (caption index, item)
    for i, items in enumerate(generated):
        seen = set()
        for item in items:
            if item["question"] not in seen:
                seen.add(item["question"])
                candidates.append((i, item))
    headings = ["(A)", "(B)", "(C)", "(D)"]
    multiple_choice = answer(
        [
            f"{item['question']} \n {item['caption']} \n "
            + "".join(f"{heading} {choice} " for heading, choice in zip(headings, item["choices"]))
            for _, item in candidates
        ]
    )
    open_ended = [
        (i, item)
        for (i, item), choice in zip(candidates, multiple_choice)
        if choice == item["answer"] and item["answer"] not in ("yes", "no")
    ]
    free_form = dict(
        zip(
            map(id, (item for _, item in open_ended)),
            answer([f"{item['question']} \n {item['caption']}" for _, item in open_ended]),
        )
    )
    kept: list[list[dict]] = [[] for _ in generated]
    for (i, item), choice in zip(candidates, multiple_choice):
        if choice != item["answer"]:
            continue
        if item["answer"] not in ("yes", "no"):
            response = "".join(c for c in free_form[id(item)] if c.isalnum() or c.isspace()).strip().lower()
            if item["answer"].isnumeric():
                try:
                    response = str(w2n.word_to_num(response))
                except Exception:  # The official filter keeps the words if they are not a number.
                    pass
            if token_f1(item["answer"].split(), response.split()) <= FREE_FORM_THRESHOLD:
                continue
        kept[i].append(item)
    return kept


def generate_questions(
    prompts: list[dict],
    output: Path,
    batch_size: int | None = None,
) -> None:
    """Write ``{"completions": {index: text}, "questions": {index: [kept items]}}`` for the prompts.

    Completions take most of the GPU time, so they are saved after each batch and reused after an interruption.
    """
    if output.exists():
        return
    if batch_size is None:  # Five beams of 512 tokens next to 13.5 GB of weights.
        memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        batch_size = 16 if memory > 80 else 8 if memory > 40 else 4
    partial = output.with_name(f".{output.stem}.partial.json")
    known = (
        {int(position): text for position, text in json.loads(partial.read_text()).items()} if partial.exists() else {}
    )
    captions = [prompt["prompt"] for prompt in prompts]
    texts = completions(captions, batch_size, known, save=lambda done: write_json(partial, done))
    kept = filter_questions([parse(caption, text) for caption, text in zip(captions, texts)])
    indices = [str(prompt["index"]) for prompt in prompts]
    write_json(output, {"completions": dict(zip(indices, texts)), "questions": dict(zip(indices, kept))})
    partial.unlink(missing_ok=True)


def write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1))
    temporary.replace(path)


def blip_answers(
    processor: BlipProcessor,
    model: BlipForQuestionAnswering,
    pixels: Tensor,
    owner: Tensor,
    questions: list[str],
) -> list[str]:
    """``BlipForQuestionAnswering.generate`` for many (image, question) pairs at once.

    ``owner[k]`` is the image of question ``k``. Question padding is masked in the answer decoder's cross-attention.
    """
    image_embeds = model.vision_model(pixel_values=pixels)[0][owner]
    query = processor.tokenizer(questions, return_tensors="pt", padding=True).to(pixels.device)
    question_embeds = model.text_encoder(
        input_ids=query.input_ids,
        attention_mask=query.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=pixels.device),
        return_dict=False,
    )[0]
    start = torch.full((len(questions), 1), model.decoder_start_token_id, device=pixels.device)
    output = model.text_decoder.generate(
        input_ids=start,
        eos_token_id=model.config.text_config.sep_token_id,
        pad_token_id=model.config.text_config.pad_token_id,
        encoder_hidden_states=question_embeds,
        encoder_attention_mask=query.attention_mask,
        max_length=50,
    )
    return processor.batch_decode(output, skip_special_tokens=True)


def sbert_embeddings(
    tokenizer: PreTrainedTokenizerBase,
    model: MPNetModel,
    sentences: list[str],
) -> Tensor:
    """``mc_sbert.SBERTModel.embed_sentences``: mean pooling over the attention mask, then L2 normalization."""
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(model.device)
    hidden = model(**tokens)[0]
    mask = tokens.attention_mask.unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)


def tifa_scorer(
    batch_size: int = 16,
) -> Callable[[list[str | Path], list[str], list[list[dict]]], list[float]]:
    """Return ``score(images, captions, questions)``, which gives the TIFA score of each image.

    ``questions[i]`` are the kept questions of ``captions[i]``.
    """
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer, BlipForQuestionAnswering, MPNetModel

    processor = AutoProcessor.from_pretrained(VQA)
    model = BlipForQuestionAnswering.from_pretrained(VQA).eval().to(device())
    sbert_tokenizer = AutoTokenizer.from_pretrained(SBERT)
    sbert = MPNetModel.from_pretrained(SBERT).eval().to(device())

    def open_rgb(
        path: str | Path,
    ) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    @torch.inference_mode()
    def score(
        images: list[str | Path],
        captions: list[str],
        questions: list[list[dict]],
    ) -> list[float]:
        scores = []
        for start in range(0, len(images), batch_size):
            chunk = range(start, min(start + batch_size, len(images)))
            asked = [(i, item) for i in chunk for item in questions[i]]
            if not asked:
                scores += [math.nan] * len(chunk)
                continue
            pixels = processor(images=[open_rgb(images[i]) for i in chunk], return_tensors="pt").pixel_values.to(
                device()
            )
            owner = torch.tensor([i - start for i, _ in asked], device=device())
            answers = blip_answers(processor, model, pixels, owner, [item["question"] for _, item in asked])
            unmatched = [
                (answer, item["choices"]) for answer, (_, item) in zip(answers, asked) if answer not in item["choices"]
            ]
            strings = list(dict.fromkeys(s for answer, choices in unmatched for s in [answer, *choices]))
            embedded = dict(zip(strings, sbert_embeddings(sbert_tokenizer, sbert, strings))) if strings else {}
            correct = defaultdict(int)
            for answer, (i, item) in zip(answers, asked):
                picked = answer
                if answer not in item["choices"]:
                    similarity = torch.stack([embedded[choice] for choice in item["choices"]]) @ embedded[answer]
                    picked = item["choices"][int(similarity.argmax())]
                correct[i] += picked == item["answer"]
            scores += [correct[i] / len(questions[i]) if questions[i] else math.nan for i in chunk]
        return scores

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prompts = [json.loads(line) for line in arguments.prompts.read_text().splitlines()][
        arguments.start : arguments.stop
    ]
    generate_questions(prompts, arguments.output)


if __name__ == "__main__":
    main()
