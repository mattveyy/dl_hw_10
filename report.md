# Report

## Track

Выбранный трек:

```text
A (CPU-only)
```

## Что реализовано

- [x] `dataset.py` — `MathVQADataset` с поддержкой split / max_samples / subjects, загрузка PIL-изображений в RGB, sanitize вопросов
- [x] `processor.py` — препроцессинг изображений (resize + ImageNet-нормализация + тайлы), prompt с `<image_start>/<image>/<image_end>`, маскирование `labels` через `IGNORE_INDEX`, паддинг в `collate`
- [x] `model.py` — `VisionToTextAdapter` (LayerNorm → Linear → GELU → Linear + adaptive pool до `num_image_tokens`), `merge_visual_embeddings`, обёртка `MathVLM` с frozen backbones
- [x] `train.py` — `load_config`, `set_seed`, `train_one_step` с проверкой конечности loss, `run_training` с gradient accumulation и сохранением чекпойнта
- [x] `benchmark.py` — `parse_mc_answer` (regex по `A/B/C/D/E`, варианты с "Answer:", "(B)", в любом окружении), `build_benchmark_prompt`, `compute_accuracy` overall/by-subject

## Конфигурация

```text
config path:      configs/track_a_cpu.yaml
seed:             42
device:           cpu
dtype:            float32
max_steps:        3 (2 при --fast-train)
batch size:       1 (global = 1, без gradient accumulation)
```

## Результаты

```text
public tests:        14 passed in ~13s (pytest -q tests_public)
train loss:          11.41 → 0.00 за 2 шага (toy surrogate model, smoke-check)
benchmark accuracy:  0.0 на toy_math_vqa/dev (используется placeholder-предсказание,
                     так как обязательный CPU-трек не требует реального LLM)
```

## Использованные ресурсы

```text
CPU/GPU:          Apple Silicon (Mac), CPU only
VRAM:             — (GPU не использовался)
время обучения:   ~1 секунда на smoke-train
```


