"""Compare our TIFA (``modalities/text_to_image/tifa.py``) with the official code (github.com/Yushi-Hu/tifa,
``tifascore``) step by step. Each step must be identical: the LLaMA-2 completions (ours batched, the official ones
one at a time), the parsed questions, the UnifiedQA filter, and the BLIP-large answers with SBERT choice matching on
images.

The official package imports clients of models that the test does not use (GPT-3.5, mPLUG, OFA, PromptCap, BLIP-2),
so they are stubs. Run ``tests/tifa/clone.sh`` first. The test needs the ``t2i-metrics`` environment, a GPU with
24 GB and the CyclePrefDB prompts. It takes about 15 minutes.
"""

import gc
import importlib.machinery
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from modalities.text_to_image import CAPTIONS
from tests.official_code import official_code
from unpaired_rosetta.embeddings import storage_root

pytest.importorskip(
    "word2number", reason="runs in the t2i-metrics environment: pixi run -e t2i-metrics pytest tests/tifa"
)

OFFICIAL = Path(__file__).parent / "official"
PROMPTS = storage_root() / "data" / "text_to_image" / "cycleprefdb" / "prompts.jsonl"
pytestmark = [
    pytest.mark.official,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not OFFICIAL.exists(), reason="run tests/tifa/clone.sh"),
    pytest.mark.skipif(not PROMPTS.exists(), reason="run python -m modalities.text_to_image.prompts"),
]
NUM_PROMPTS = 16
UNUSED_CLIENTS = {
    "openai": [],
    "modelscope": [],
    "modelscope.pipelines": ["pipeline"],
    "modelscope.utils": [],
    "modelscope.utils.constant": ["Tasks"],
    "modelscope.outputs": ["OutputKeys"],
    "modelscope.preprocessors": [],
    "modelscope.preprocessors.multi_modal": ["OfaPreprocessor"],
    "promptcap": ["PromptCap_VQA"],
    "lavis": [],
    "lavis.models": ["load_model_and_preprocess"],
}


@pytest.fixture(scope="module")
def official(
    monkeypatch_module: pytest.MonkeyPatch,
) -> Iterator[types.SimpleNamespace]:
    for name, attributes in UNUSED_CLIENTS.items():
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for attribute in attributes:
            setattr(stub, attribute, None)
        monkeypatch_module.setitem(sys.modules, name, stub)
    with official_code(OFFICIAL):
        import tifascore
        from tifascore import mc_sbert, question_filter, question_gen_llama2

        yield types.SimpleNamespace(
            package=tifascore, generation=question_gen_llama2, filter=question_filter, sbert=mc_sbert
        )


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


@pytest.fixture(scope="module")
def prompts() -> list[dict[str, str]]:
    return [json.loads(line) for line in PROMPTS.read_text().splitlines()[:NUM_PROMPTS]]


@pytest.fixture(scope="module")
def captions(
    prompts: list[dict[str, str]],
) -> list[str]:
    return list(CAPTIONS) + [prompt["prompt"] for prompt in prompts]


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def official_completions(
    official: types.SimpleNamespace,
    captions: list[str],
) -> list[str]:
    pipeline = official.generation.get_llama2_pipeline()
    completions = [official.generation.llama2_completion(pipeline, caption) for caption in captions]
    del pipeline
    free_gpu()
    return completions


@pytest.fixture(scope="module")
def official_questions(
    official: types.SimpleNamespace,
    captions: list[str],
    official_completions: list[str],
) -> list[list[dict[str, object]]]:
    parsed = []
    for caption, completion in zip(captions, official_completions):
        official.generation.llama2_completion = lambda pipeline, caption, completion=completion: completion
        parsed.append(official.generation.get_llama2_question_and_answers(None, caption))
    return parsed


def test_completions(
    captions: list[str],
    official_completions: list[str],
) -> None:
    from modalities.text_to_image.tifa import completions

    ours = completions(captions, batch_size=4, known={})
    free_gpu()
    print(f"TIFA completions: {sum(a == b for a, b in zip(ours, official_completions))} of {len(captions)} identical")
    assert ours == official_completions


def test_parsed_questions(
    captions: list[str],
    official_completions: list[str],
    official_questions: list[list[dict[str, object]]],
) -> None:
    from modalities.text_to_image.tifa import parse

    ours = [parse(caption, completion) for caption, completion in zip(captions, official_completions)]
    print(f"TIFA parsed questions: {sum(map(len, ours))}")
    assert ours == official_questions


def test_filter(
    official: types.SimpleNamespace,
    official_questions: list[list[dict[str, object]]],
) -> None:
    from modalities.text_to_image.tifa import FILTER, filter_questions

    answerer = official.package.UnifiedQAModel(FILTER)
    kept = [official.filter.filter_question_and_answers(answerer, items) for items in official_questions]
    del answerer
    free_gpu()
    ours = filter_questions(official_questions)
    free_gpu()
    print(f"TIFA kept questions: {sum(map(len, ours))} of {sum(map(len, official_questions))}")
    assert ours == kept


def test_answers_on_images(
    official: types.SimpleNamespace,
    prompts: list[dict[str, str]],
    captions: list[str],
    official_questions: list[list[dict[str, object]]],
) -> None:
    """The answers agree for each photograph with its own prompt and with the prompt of the next photograph."""
    from transformers import MPNetModel

    from modalities.text_to_image.tifa import tifa_scorer

    by_caption = dict(zip(captions, official_questions))
    images = [prompt["image"] for prompt in prompts] * 2
    image_captions = [prompt["prompt"] for prompt in prompts] + [
        prompts[(i + 1) % NUM_PROMPTS]["prompt"] for i in range(NUM_PROMPTS)
    ]
    questions = [by_caption[caption] for caption in image_captions]
    ours = tifa_scorer()(images, image_captions, questions)
    free_gpu()
    official.sbert.AutoModel = MPNetModel  # AutoModel fails with the old timm of this environment
    model = official.package.VQAModel("blip-large")
    theirs = []
    for image, items in zip(images, questions):
        hits = [
            model.multiple_choice_vqa(image, item["question"], choices=item["choices"])["multiple_choice_answer"]
            == item["answer"]
            for item in items
        ]
        theirs.append(sum(hits) / len(hits) if hits else float("nan"))
    identical = sum(a == b or (a != a and b != b) for a, b in zip(ours, theirs))
    print(f"TIFA scores: {identical} of {len(images)} identical, {sum(map(len, questions))} questions")
    assert identical == len(images)
