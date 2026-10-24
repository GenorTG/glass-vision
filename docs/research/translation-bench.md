# Translation benchmark — signs, menus and notices (2026-10-24)

Reproducible benchmark assets and raw outputs: [`translation-bench/`](translation-bench/). Method and reproduction steps are in its README.

## Result

Across 259 curated items in 15 source languages (targets: English, Polish, German and Spanish), the strongest overall system was **Index-Translate-9B Q4_K_M with the sign-aware prompt**. It scored COMET 0.933, chrF++ 76.5, and preserved every tested line break. The current M2M100-1.2B block mode scored COMET 0.792, chrF++ 45.8, and preserved line breaks, while the 418M block mode scored COMET 0.791 / chrF++ 47.5. These are benchmark metrics, not a claim that every individual translation is better.

| System | COMET | chrF++ | Sign key-term hit | Line structure kept | p50 / p95 latency | VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Index-Translate-9B Q4_K_M, sign prompt | **0.933** | **76.5** | **94%** | **100%** | 142 / 458 ms | 5,680 MiB |
| Gemma 4 E4B Q4_K_M, sign prompt | 0.929 | 75.4 | 92% | 100% | 257 / 523 ms | 3,702 MiB |
| Gemma 4 E2B Q8_0, sign prompt | 0.922 | 75.7 | 88% | 100% | 91 / 327 ms | 2,958 MiB |
| Qwen3.5-4B Q8_0, sign prompt | 0.919 | 71.2 | 90% | 100% | 102 / 356 ms | 5,106 MiB |
| M2M100 1.2B, block mode | 0.792 | 45.8 | 58% | 100% | 271 / 718 ms | 2,806 MiB |
| M2M100 418M, block mode | 0.791 | 47.5 | 56% | 100% | 128 / 317 ms | 1,356 MiB |
| MADLAD-400 3B, block mode | 0.851 | 57.0 | 63% | 100% | 519 / 1,602 ms | 6,358 MiB |

“Key-term hit” is a curated adequacy proxy; chrF++ is harsh on valid alternative wording. COMET is a learned metric, not human judgment. The blind-review sheet generator is included, but no human ratings have been collected yet.

## Recommendation

Keep M2M100 as the installed, self-contained fallback for now. The benchmark justifies a deployment prototype of **Index-Translate-9B behind a local OpenAI-compatible endpoint**, using the sign-aware prompt and per-block translation. Gemma 4 E2B is the lower-memory/latency candidate if model footprint wins over the small aggregate score gap. Do not select a runtime default from these scores alone: the benchmark ran standalone llama.cpp/Transformers processes and did not test service integration, cold-start behavior under the live OCR workload, language coverage beyond the four target languages, or human preference.

Next evaluation should include a blind human review of the generated 40-item sheet, then an A/B run through the actual server translation interface. The benchmark has not changed production translation behavior.

## Limitations

- Test references and the set were authored by one LLM-assisted evaluator; there was no independent human translator review. Japanese, Korean and Arabic references are explicitly the least certain in the README.
- Several systems use task-specific prompts; the results compare deployable configurations, not only model weights.
- Latency is one-host, sequential warmed inference on an RTX 3090. It excludes network and service queue time; p95 is reported because context-heavy blocks have a long tail.
- Current M2M100 block-vs-line results are effectively tied; block translation's primary product value is preserving context and sentence structure, not a measured score improvement in this test.
