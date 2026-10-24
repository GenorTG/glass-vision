"""OCR quality eval for "Read text" (docs/OCR-RESEARCH.md P0): rendered signs + real photos → a metrics table.

    ./run tools/ocr_eval.py                          # read every image with TextPipeline.read_still, print the table
    ./run tools/ocr_eval.py --json out.json          # + raw readings and metrics
    ./run tools/ocr_eval.py --compare before.json    # + the same table for an earlier run (e.g. the EasyOCR baseline)
    ./run tools/ocr_eval.py --only render_de_rot180 --show   # print what was read

Also the library behind `tests/test_text_read_quality.py`. Engine-agnostic: it only looks at the `lines` of a
`read_still` response (`text`, `box`), so any engine can be compared on the same set.

The set (ideas from docs/research/bench/scripts/make_signs.py, regenerated here deterministically):
- rendered signs on a noisy wall: German 3-line sign, the same at 90/180/270° and perspective-skewed, a Polish 6-line
  paragraph straight and tilted 15°, Russian, Arabic, Chinese, Japanese and Greek signs (no Hangul font on the
  host for Korean); ground truth = lines;
- real photos: ultralytics' bus.jpg + 9 Wikimedia Commons photos (docs/research/bench/SOURCES.tsv). They are fetched
  on first use into `tests/data/ocr/` (gitignored, ~35 MB of originals; sha1 checked against Commons) instead of being
  committed; attribution: `tests/data/ocr/README.md`. Offline, the photos are skipped. Ground truth = key phrases.

Metrics (the bench's, docs/research/ocr-server-engines.md §3):
- key recall (photos): share of key phrases found with a fuzzy partial match >= 0.9 (edit distance against the best
  substring of the whole reading, after NFKC, case-folding and removing whitespace);
- lines exact (rendered): ground-truth lines read by ONE output line with CER <= 5 %;
- boxes / GT line (rendered): output lines per ground-truth line (word fragments push it above 1);
- order ok (rendered): the exactly-read lines come out in reading order;
- rotated ok: rotated/skewed/tilted signs with every line exact;
- latency: `ms.ocr` of the response (median of the photos, after one warm-up read per image).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import unicodedata
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SERVER = Path(__file__).resolve().parents[1]
DATA = SERVER / "tests" / "data" / "ocr"
FONTS = Path("/usr/share/fonts/truetype")
BOLD = FONTS / "dejavu/DejaVuSans-Bold.ttf"
REGULAR = FONTS / "dejavu/DejaVuSans.ttf"
CJK = FONTS / "droid/DroidSansFallbackFull.ttf"
KEY_MATCH = 0.9
LINE_CER = 0.05
USER_AGENT = "glass-vision-ocr-eval/1 (https://github.com/; test images, fetched once)"


@dataclass(frozen=True)
class Photo:
    stem: str
    url: str  # Commons original
    sha1: str  # Commons imageinfo sha1
    license: str
    author: str
    page: str
    keys: tuple[str, ...]


# fmt: off
COMMONS = "https://upload.wikimedia.org/wikipedia/commons"
PHOTOS: tuple[Photo, ...] = (
    Photo("Emergency_exit_sign_01", f"{COMMONS}/6/6a/Emergency_exit_sign_01.jpg",
          "9dff07eedeb0266602b00e8f31d78fb9312689e5", "CC BY-SA 4.0", "Kritzolina",
          "https://commons.wikimedia.org/wiki/File:Emergency_exit_sign_01.jpg", ("Notausgang", "freihalten")),
    Photo("Taiwan_exit_sign", f"{COMMONS}/8/8b/Taiwan_exit_sign_white_backgrounded.jpg",
          "3adcb040d8c11cafc94d8d8b35ccf612e9751b0b", "Public domain", "Oakw",
          "https://commons.wikimedia.org/wiki/File:Taiwan_exit_sign_white_backgrounded.jpg", ("緊急出口", "EXIT")),
    Photo("Tokyo_Station_sign", f"{COMMONS}/b/b7/Tokyo_Station_Sign_%28Yamanote_Line%29.jpg",
          "418270aa2c312bc68cfda3d47acaf2b93cd0ef11", "CC BY-SA 3.0", "そらみみ",
          "https://commons.wikimedia.org/wiki/File:Tokyo_Station_Sign_(Yamanote_Line).jpg",
          ("東京", "とうきょう", "有楽町", "Yūrakuchō", "神田", "Kanda", "Tōkyō", "山区")),
    Photo("Moscow_Arbatskaya_sign",
          f"{COMMONS}/0/06/Moscow%2C_Arbatskaya_%28APL%29_metro_station_-_sign_%2843315108441%29.jpg",
          "3ed61f83d01e1ac957d2e5a2aabfdb455285eb36", "CC BY 2.0", "Andrey Filippov",
          "https://commons.wikimedia.org/wiki/File:Moscow,_Arbatskaya_(APL)_metro_station_-_sign_(43315108441).jpg",
          ("АРБАТСКАЯ",)),
    Photo("Sidi_El_Wasti_sign",
          f"{COMMONS}/5/5e/Arabic-English_street_sign_%22Sidi_El_Wasti_St.%22_in_Alexandria.jpg",
          "7d0eded8bc7ba747d8b6834076cc149ee6702c2b", "CC BY 4.0", "Marsupium",
          "https://commons.wikimedia.org/wiki/File:Arabic-English_street_sign_%22Sidi_El_Wasti_St.%22_in_Alexandria.jpg",
          ("شارع سيدى الوسطى", "SIDI EL WASTI ST.", "N-P 21131", "الرقم البريدى", "٢١١٣١")),
    Photo("Del_Taco_menu", f"{COMMONS}/7/79/Drive_thru_menu_board_at_Del_Taco_in_Albany%2C_Oregon.jpg",
          "922c00d67d01faf46e5352b87561150193013cbe", "CC BY 4.0", "Rick Obst",
          "https://commons.wikimedia.org/wiki/File:Drive_thru_menu_board_at_Del_Taco_in_Albany,_Oregon.jpg",
          ("DEL TACO", "SLOW-COOKED", "CARNITAS", "STREET TACOS", "TACOS", "EPIC BURRITOS", "CLASSIC BURRITOS",
           "BURGERS", "CRINKLE CUT FRIES", "QUESADILLAS", "NACHOS & SALADS", "DRINKS, SHAKES & DESSERTS", "20% OFF",
           "INDEPENDENCE POPPERS", "BREAKFAST", "BIG FAT TACOS", "CRUNCHTADA", "Please Present Coupons",
           "FUNNEL CAKE FRIES")),
    Photo("HK_paper_menus", f"{COMMONS}/0/0c/Paper_menus_and_white_board_menu_in_the_hk_style_restaurants.jpg",
          "eac3f5a5ee313013aedc12caa1871faf3a81e8ec", "CC BY-SA 4.0", "Peachyeung316",
          "https://commons.wikimedia.org/wiki/File:Paper_menus_and_white_board_menu_in_the_hk_style_restaurants.jpg",
          ("家常便餸", "即叫即炒", "外賣自取", "電話24318638", "星期一", "星期二", "星期三", "星期四", "星期五", "奉送例湯",
           "$39快餐", "每款60元", "翠湖茶餐廳", "$46凍飲加2元")),
    Photo("Piwna_Street", f"{COMMONS}/4/47/Piwna_Street%2C_Warsaw_05.jpg",
          "bd4cf166ee45c08088b4fe079d21c60955e52220", "CC BY-SA 4.0", "Simon Burchell",
          "https://commons.wikimedia.org/wiki/File:Piwna_Street,_Warsaw_05.jpg", ("DOM KSIĄŻKI",)),
    Photo("Laptop_code", f"{COMMONS}/5/50/Macro_laptop_coding_%28Unsplash%29.jpg",
          "8b5366b9b9a81bd095e8aefa745cda06334f0f78", "CC0", "Marc Mueller",
          "https://commons.wikimedia.org/wiki/File:Macro_laptop_coding_(Unsplash).jpg",
          ("Sublime Text", "File", "Edit", "Selection", "main.js", "function hexSearch() {",
           "var re = /[0-9A-Fa-f]{6}/g;", "var re2 = /[0-9A-Fa-f]{3}/g;", "} else {", "return;",
           "var color0 = '#7822C8';", "var color1 = '#60FFFF';", "colorOne = color0;", "colorTwo = color1;")),
)
BUS_KEYS = ("cero emisiones",)
# fmt: on


@dataclass
class Sample:
    stem: str
    image: np.ndarray  # BGR
    keys: tuple[str, ...] = ()  # photos
    lines: tuple[str, ...] = ()  # rendered: ground-truth lines in reading order
    rotated: bool = False  # counts towards "rotated ok"
    kind: str = "photo"


# --- rendered signs -------------------------------------------------------------------------------------------------
def _sign(lines: list[tuple[str, int, Path]], bg: tuple[int, int, int], fg: tuple[int, int, int], rtl: bool = False):
    im = Image.new("RGB", (1400, 800), bg)
    d = ImageDraw.Draw(im)
    y = 60
    for text, pt, font in lines:
        f = ImageFont.truetype(str(font), pt)
        x = 70
        if rtl:  # right-aligned, as RTL signs are
            x = 1400 - 70 - int(d.textlength(text, font=f, direction="rtl"))
            d.text((x, y), text, font=f, fill=fg, direction="rtl")
        else:
            d.text((x, y), text, font=f, fill=fg)
        y += int(pt * 1.35)
    d.rectangle([20, 20, 1380, 780], outline=fg, width=8)
    return np.array(im)[:, :, ::-1]


def _on_wall(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paste on a noisy 2400x1800 wall, uneven light, blur + noise, then a JPEG round trip."""
    h, w = 1800, 2400
    wall = rng.normal(150, 18, (h, w, 3)).clip(0, 255).astype(np.uint8)
    wall = cv2.GaussianBlur(wall, (7, 7), 0)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    wall[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    grad = np.linspace(0.65, 1.1, w)[None, :, None]
    out = (wall * grad + rng.normal(0, 6, wall.shape)).clip(0, 255).astype(np.uint8)
    return _jpeg_round_trip(cv2.GaussianBlur(out, (3, 3), 0))


def _jpeg_round_trip(img: np.ndarray, quality: int = 88) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def _perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.12, h * 0.05], [w * 0.95, h * 0.22], [w * 0.88, h * 0.80], [w * 0.05, h * 0.97]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(140, 140, 140))


def _tilt(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(140, 140, 140))


# fmt: off
DE = [("NOTAUSGANG", 150, BOLD), ("Bitte freihalten", 90, REGULAR), ("Abfahrt 12:45 · Gleis 3", 80, REGULAR)]
PL = [("Szanowni Państwo,", 62, BOLD), ("w związku z remontem wejście", 62, REGULAR),
      ("główne jest nieczynne do", 62, REGULAR), ("30 listopada. Prosimy korzystać", 62, REGULAR),
      ("z wejścia od ulicy Piwnej.", 62, REGULAR), ("Dziękujemy za wyrozumiałość!", 62, REGULAR)]
RU = [("ВЫХОД В ГОРОД", 120, BOLD), ("к улице Арбат", 90, REGULAR), ("Не загораживать!", 80, REGULAR)]
AR = [("مخرج الطوارئ", 140, BOLD), ("ممنوع التدخين", 110, REGULAR), ("شارع النصر", 100, REGULAR)]
ZH = [("紧急出口", 170, CJK), ("请勿堵塞通道", 120, CJK), ("出口在左侧", 110, CJK)]
JA = [("非常口", 170, CJK), ("とうきょう駅", 120, CJK), ("足元に注意", 110, CJK)]
EL = [("ΕΞΟΔΟΣ ΚΙΝΔΥΝΟΥ", 120, BOLD), ("Απαγορεύεται το κάπνισμα", 76, REGULAR), ("Οδός Ερμού", 90, REGULAR)]
# fmt: on


def rendered_samples() -> list[Sample]:
    """Deterministic (fixed seed): the same pixels on every run."""
    rng = np.random.default_rng(7)

    def make(lines, bg, fg, rtl=False) -> np.ndarray:
        return _on_wall(_sign(lines, bg, fg, rtl), rng)

    def gt(lines) -> tuple[str, ...]:
        return tuple(t for t, _, _ in lines)

    de = make(DE, (40, 120, 40), (255, 255, 255))
    pl = make(PL, (245, 245, 235), (20, 20, 20))
    plain = {
        "render_de": (de, DE),
        "render_pl_paragraph": (pl, PL),
        "render_ru": (make(RU, (180, 60, 20), (255, 255, 255)), RU),
        "render_ar": (make(AR, (30, 90, 30), (255, 255, 255), rtl=True), AR),
        "render_zh": (make(ZH, (30, 130, 30), (255, 255, 255)), ZH),
        "render_ja": (make(JA, (250, 250, 250), (20, 20, 20)), JA),
        "render_el": (make(EL, (120, 60, 20), (255, 255, 255)), EL),
    }
    rotated = {
        "render_de_rot90": (cv2.rotate(de, cv2.ROTATE_90_CLOCKWISE), DE),
        "render_de_rot180": (cv2.rotate(de, cv2.ROTATE_180), DE),
        "render_de_rot270": (cv2.rotate(de, cv2.ROTATE_90_COUNTERCLOCKWISE), DE),
        "render_de_persp": (_jpeg_round_trip(_perspective(de)), DE),
        "render_pl_tilt15": (_jpeg_round_trip(_tilt(pl, 15)), PL),
    }
    out = [Sample(stem, image, lines=gt(lines), kind="render") for stem, (image, lines) in plain.items()]
    out += [
        Sample(stem, image, lines=gt(lines), rotated=True, kind="render") for stem, (image, lines) in rotated.items()
    ]
    return out


# --- photos ---------------------------------------------------------------------------------------------------------
def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def fetch_photo(photo: Photo, data: Path = DATA) -> Path | None:
    """The cached original, downloading it once; None when offline (or Commons changed the file)."""
    target = data / f"{photo.stem}.jpg"
    if target.is_file():
        return target
    data.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    try:
        request = urllib.request.Request(photo.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            part.write_bytes(response.read())
    except OSError as exc:
        print(f"[ocr-eval] {photo.stem}: not available ({exc}); skipped", file=sys.stderr)
        return None
    if _sha1(part) != photo.sha1:
        part.unlink()
        print(f"[ocr-eval] {photo.stem}: sha1 mismatch (changed on Commons?); skipped", file=sys.stderr)
        return None
    part.rename(target)
    return target


def photo_samples(data: Path = DATA) -> list[Sample]:
    from ultralytics.utils import ASSETS

    out = [Sample("bus", cv2.imread(str(ASSETS / "bus.jpg")), keys=BUS_KEYS)]
    for photo in PHOTOS:
        path = fetch_photo(photo, data)
        if path is not None:
            out.append(Sample(photo.stem, cv2.imread(str(path), cv2.IMREAD_COLOR), keys=photo.keys))
    return out


def all_samples(photos: bool = True) -> list[Sample]:
    return rendered_samples() + (photo_samples() if photos else [])


# --- metrics --------------------------------------------------------------------------------------------------------
def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(truth: str, read: str) -> float:
    return levenshtein(truth, read) / max(len(truth), len(read), 1)


def partial_match(key: str, text: str) -> float:
    """1 - (edit distance of `key` against its best-matching substring of `text`) / len(key) (Sellers)."""
    if not key:
        return 1.0
    if not text:
        return 0.0
    m = len(key)
    column = list(range(m + 1))  # D[i] for the text prefix so far; D[0] = 0 at every text position
    best = column[m]
    for ch in text:
        previous_diagonal, column[0] = column[0], 0
        for i in range(1, m + 1):
            current = min(column[i] + 1, column[i - 1] + 1, previous_diagonal + (key[i - 1] != ch))
            previous_diagonal, column[i] = column[i], current
        best = min(best, column[m])
    return 1 - best / m


@dataclass
class Score:
    stem: str
    kind: str
    ms: float
    angle: float | None
    n_lines: int
    keys_hit: int = 0
    keys_total: int = 0
    lines_exact: int = 0
    lines_total: int = 0
    order_ok: bool = True
    rotated: bool = False
    texts: list[str] = field(default_factory=list)

    @property
    def all_exact(self) -> bool:
        return self.lines_total > 0 and self.lines_exact == self.lines_total


def score(sample: Sample, response: dict[str, Any]) -> Score:
    texts = [line["text"] for line in response["lines"] if line["text"].strip()]
    out = Score(
        sample.stem,
        sample.kind,
        float(response["ms"]["ocr"]),
        response.get("angle"),
        len(texts),
        rotated=sample.rotated,
        texts=texts,
    )
    if sample.keys:
        joined = norm("".join(texts))
        out.keys_total = len(sample.keys)
        out.keys_hit = sum(partial_match(norm(k), joined) >= KEY_MATCH for k in sample.keys)
    if sample.lines:
        reads = [norm(t) for t in texts]
        positions = []
        for truth in sample.lines:
            hits = [i for i, r in enumerate(reads) if cer(norm(truth), r) <= LINE_CER]
            if hits:
                positions.append(hits[0])
        out.lines_total = len(sample.lines)
        out.lines_exact = len(positions)
        out.order_ok = positions == sorted(positions)
    return out


ReadFn = Callable[[np.ndarray], dict[str, Any]]


def evaluate(read: ReadFn, samples: Iterable[Sample], warm: bool = True, show: bool = False) -> list[Score]:
    scores = []
    for sample in samples:
        if warm:
            read(sample.image)  # loads script recognizers etc. once; the timed read is the second one
        t0 = time.perf_counter()
        response = read(sample.image)
        wall = (time.perf_counter() - t0) * 1000
        s = score(sample, response)
        scores.append(s)
        if show:
            print(f"--- {sample.stem} ({s.ms:.0f} ms ocr, {wall:.0f} ms wall, angle {s.angle})")
            for line in response["lines"]:
                print(f"    {line.get('block')}: {line['text']!r} {line.get('conf')}")
    return scores


@dataclass
class Summary:
    keys_hit: int
    keys_total: int
    lines_exact: int
    lines_total: int
    boxes_per_line: float
    order_ok: int
    rendered: int
    rotated_ok: int
    rotated_total: int
    median_ms_photos: float
    max_ms_photos: float
    median_ms_rendered: float

    @property
    def key_recall(self) -> float:
        return self.keys_hit / self.keys_total if self.keys_total else 0.0


def summarize(scores: list[Score]) -> Summary:
    photos = [s for s in scores if s.kind == "photo"]
    rendered = [s for s in scores if s.kind == "render"]
    lines_total = sum(s.lines_total for s in rendered)
    return Summary(
        keys_hit=sum(s.keys_hit for s in photos),
        keys_total=sum(s.keys_total for s in photos),
        lines_exact=sum(s.lines_exact for s in rendered),
        lines_total=lines_total,
        boxes_per_line=sum(s.n_lines for s in rendered) / lines_total if lines_total else 0.0,
        order_ok=sum(s.order_ok and s.all_exact for s in rendered),
        rendered=len(rendered),
        rotated_ok=sum(s.all_exact for s in rendered if s.rotated),
        rotated_total=sum(s.rotated for s in rendered),
        median_ms_photos=statistics.median(s.ms for s in photos) if photos else 0.0,
        max_ms_photos=max((s.ms for s in photos), default=0.0),
        median_ms_rendered=statistics.median(s.ms for s in rendered) if rendered else 0.0,
    )


def table(runs: dict[str, list[Score]]) -> str:
    """Markdown: one summary row per run, then one row per image."""
    out = [
        (
            "| run | photo key recall | rendered lines exact | boxes / GT line | all lines exact + in order "
            "| rotated/skewed ok | median ms photos (max) | median ms rendered |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, scores in runs.items():
        s = summarize(scores)
        out.append(
            f"| {name} | {s.keys_hit}/{s.keys_total} ({s.key_recall:.0%}) | {s.lines_exact}/{s.lines_total} "
            f"| {s.boxes_per_line:.2f} | {s.order_ok}/{s.rendered} | {s.rotated_ok}/{s.rotated_total} "
            f"| {s.median_ms_photos:.0f} ({s.max_ms_photos:.0f}) | {s.median_ms_rendered:.0f} |"
        )
    stems = list(dict.fromkeys(s.stem for scores in runs.values() for s in scores))
    out += ["", "| image | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for stem in stems:
        cells = []
        for scores in runs.values():
            s = next((x for x in scores if x.stem == stem), None)
            if s is None:
                cells.append("")
            elif s.kind == "photo":
                cells.append(f"{s.keys_hit}/{s.keys_total} keys · {s.ms:.0f} ms")
            else:
                order = "" if s.order_ok else " (order!)"
                cells.append(f"{s.lines_exact}/{s.lines_total} lines{order} · {s.n_lines} boxes · {s.ms:.0f} ms")
        out.append(f"| {stem} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def scores_to_json(scores: list[Score]) -> list[dict[str, Any]]:
    return [s.__dict__ | {"all_exact": s.all_exact} for s in scores]


def scores_from_json(items: list[dict[str, Any]]) -> list[Score]:
    return [Score(**{k: v for k, v in item.items() if k != "all_exact"}) for item in items]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", type=Path, help="write scores (+ readings) here")
    parser.add_argument("--name", default="current", help="row name in the table")
    parser.add_argument("--compare", type=Path, action="append", default=[], help="earlier --json output(s)")
    parser.add_argument("--only", help="comma-separated stems")
    parser.add_argument("--no-photos", action="store_true")
    parser.add_argument("--show", action="store_true", help="print every reading")
    parser.add_argument("--languages", help="text pipeline `languages` value (default: its default)")
    args = parser.parse_args()

    sys.path.insert(0, str(SERVER))
    from glass_vision.pipelines.text import TextPipeline

    samples = all_samples(photos=not args.no_photos)
    if args.only:
        wanted = set(args.only.split(","))
        samples = [s for s in samples if s.stem in wanted]
    pipeline = TextPipeline()
    t0 = time.perf_counter()
    pipeline.load(args.device)
    print(f"[ocr-eval] loaded in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    languages = args.languages or next(o.default for o in TextPipeline.info.options if o.key == "languages")
    scores = evaluate(lambda image: pipeline.read_still(image, languages), samples, show=args.show)
    runs = {path.stem: scores_from_json(json.loads(path.read_text())["scores"]) for path in args.compare}
    runs[args.name] = scores
    print(table(runs))
    if args.json:
        args.json.write_text(json.dumps({"name": args.name, "scores": scores_to_json(scores)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
