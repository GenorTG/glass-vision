# OCR eval results (docs/OCR-RESEARCH.md P0)

Set and metrics: `server/tools/ocr_eval.py` (12 rendered signs incl. 90/180/270°, perspective, 15° tilt, Cyrillic,
Arabic, Chinese, Japanese, Greek; bus.jpg + 9 Wikimedia Commons photos, attribution in
`server/tests/data/ocr/README.md`). Regression floors: `server/tests/test_text_read_quality.py`.

Re-run and compare: `cd server && ./run tools/ocr_eval.py --compare ../docs/research/ocr-eval/easyocr_baseline.json`.

## EasyOCR baseline (2026-10-24, v0.6, RTX 3090 shared with the live service, auto script mode, all 12 script groups installed)

`easyocr_baseline.json` (scores + every reading).

| run | photo key recall | rendered lines exact | boxes / GT line | all lines exact + in order | rotated/skewed ok | median ms photos (max) | median ms rendered |
|---|---|---|---|---|---|---|---|
| easyocr_baseline | 28/67 (42%) | 15/42 | 1.52 | 1/12 | 0/5 | 901 (66829) | 620 |

| image | easyocr_baseline |
|---|---|
| render_de | 2/3 lines · 4 boxes · 411 ms |
| render_de_rot90 | 0/3 lines · 2 boxes · 565 ms |
| render_de_rot180 | 0/3 lines · 2 boxes · 512 ms |
| render_de_rot270 | 0/3 lines · 2 boxes · 689 ms |
| render_de_persp | 2/3 lines · 4 boxes · 412 ms |
| render_pl_paragraph | 5/6 lines · 10 boxes · 1916 ms |
| render_pl_tilt15 | 0/6 lines · 23 boxes · 960 ms |
| render_ru | 1/3 lines · 4 boxes · 448 ms |
| render_ar | 2/3 lines · 4 boxes · 664 ms |
| render_zh | 0/3 lines · 3 boxes · 618 ms |
| render_ja | 3/3 lines · 3 boxes · 622 ms |
| render_el | 0/3 lines · 3 boxes · 890 ms |
| bus | 1/1 keys · 870 ms |
| Emergency_exit_sign_01 | 0/2 keys · 380 ms |
| Taiwan_exit_sign | 0/2 keys · 213 ms |
| Tokyo_Station_sign | 5/8 keys · 932 ms |
| Moscow_Arbatskaya_sign | 0/1 keys · 144 ms |
| Sidi_El_Wasti_sign | 1/5 keys · 1177 ms |
| Del_Taco_menu | 10/19 keys · 66829 ms |
| HK_paper_menus | 5/14 keys · 18368 ms |
| Piwna_Street | 0/1 keys · 734 ms |
| Laptop_code | 6/14 keys · 9934 ms |
