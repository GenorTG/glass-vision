"""Open-vocabulary detection: the user types what to look for ("red mug, keys, glasses") — YOLOE-11s.

YOLOE bakes the text prompt into the model (`set_classes` with MobileCLIP text embeddings) and drops its
predictor when the prompt changes, so the next predict re-fuses the model (~250 ms). Instead of swapping
prompts on one instance, keep a ready instance per recent prompt (LRU of MAX_CACHED_PROMPTS, ~100 MB VRAM
each): switching back to a recent prompt is free, a newly typed one costs ~0.5 s once.

Needs `MODELS_DIR / mobileclip_blt.ts` (570 MB, see CLIP_WEIGHTS) next to the YOLOE weights.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from ultralytics import YOLOE
from ultralytics.nn.text_model import MobileCLIPTS

from ._ultralytics import WARMUP_IMAGE, predict_kwargs, to_annotations
from .base import MODELS_DIR, Annotation, OptionSpec, Pipeline, PipelineInfo, RunContext

WEIGHTS = "yoloe-11s-seg.pt"
CLIP_WEIGHTS = "mobileclip_blt.ts"  # https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip_blt.ts
DEFAULT_PROMPT = "person, cup, phone, keys, glasses"
MAX_CLASSES = 20
MAX_CACHED_PROMPTS = 3

Prompt = tuple[str, ...]


def parse_prompt(prompt: str) -> Prompt:
    names = [p.strip().lower() for p in prompt.replace(";", ",").replace("\n", ",").split(",")]
    unique = list(dict.fromkeys(n for n in names if n))
    return tuple(unique[:MAX_CLASSES])


class OpenVocabPipeline(Pipeline):
    info = PipelineInfo(
        id="open_vocab",
        name="Find by text",
        description="Type what to look for, comma-separated — anything you can describe in a few words (YOLOE).",
        uses_global_conf=True,
        options=[OptionSpec("prompt", "What to look for (comma-separated)", "text", DEFAULT_PROMPT)],
    )

    def __init__(self) -> None:
        self._device = "cpu"
        self._encoder: YOLOE | None = None  # only computes text embeddings
        self._models: OrderedDict[Prompt, YOLOE] = OrderedDict()

    def load(self, device: str) -> None:
        self._device = device
        self._encoder = YOLOE(str(MODELS_DIR / WEIGHTS))
        self._encoder.to(device)  # get_text_pe runs on the model's own device
        # Hand YOLOE its text encoder explicitly: left alone it re-loads MobileCLIP from the *cwd* (downloading
        # it there if missing) on every get_text_pe call.
        self._encoder.model.clip_model = MobileCLIPTS(torch.device(device), weight=str(MODELS_DIR / CLIP_WEIGHTS))
        self._model_for(parse_prompt(DEFAULT_PROMPT))

    def unload(self) -> None:
        self._encoder = None
        self._models.clear()

    def _model_for(self, names: Prompt) -> YOLOE:
        model = self._models.get(names)
        if model is not None:
            self._models.move_to_end(names)
            return model
        assert self._encoder is not None, "not loaded"
        embeddings = self._encoder.model.get_text_pe(list(names), cache_clip_model=True)
        model = YOLOE(str(MODELS_DIR / WEIGHTS))
        model.to(self._device)
        model.set_classes(list(names), embeddings)
        model.predict(WARMUP_IMAGE, device=self._device, verbose=False)  # builds + fuses the predictor now
        self._models[names] = model
        while len(self._models) > MAX_CACHED_PROMPTS:
            self._models.popitem(last=False)
        return model

    def run(self, image: np.ndarray, ctx: RunContext) -> list[Annotation]:
        names = parse_prompt(str(ctx.options.get("prompt", DEFAULT_PROMPT)))
        if not names:
            return []
        result = self._model_for(names).predict(image, **predict_kwargs(ctx, self._device))[0]
        result.masks = None  # segmentation masks aren't sent; skip the CPU copy
        return to_annotations(result)
