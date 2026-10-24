"""Render synthetic signs (multi-line, paragraph, Cyrillic) + rotated / perspective copies, with ground truth."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(sys.argv[1])
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
rng = np.random.default_rng(7)


def sign(lines, bg, fg, size=(1400, 800)):
    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    y = 60
    for text, pt, font in lines:
        f = ImageFont.truetype(font, pt)
        d.text((70, y), text, font=f, fill=fg)
        y += int(pt * 1.35)
    d.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=fg, width=8)
    return np.array(im)[:, :, ::-1]


def on_wall(img, canvas=(2400, 1800)):
    """Paste on a noisy wall, add blur/noise/uneven light, JPEG-ish."""
    h, w = canvas[1], canvas[0]
    wall = (rng.normal(150, 18, (h, w, 3))).clip(0, 255).astype(np.uint8)
    wall = cv2.GaussianBlur(wall, (7, 7), 0)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    wall[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    grad = np.linspace(0.65, 1.1, w)[None, :, None]
    out = (wall * grad + rng.normal(0, 6, wall.shape)).clip(0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (3, 3), 0)


def perspective(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.12, h * 0.05], [w * 0.95, h * 0.22], [w * 0.88, h * 0.80], [w * 0.05, h * 0.97]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(140, 140, 140))


signs = {
    "render_de": (
        [("NOTAUSGANG", 150, BOLD), ("Bitte freihalten", 90, REG), ("Abfahrt 12:45 · Gleis 3", 80, REG)],
        (40, 120, 40), (255, 255, 255),
    ),
    "render_pl_paragraph": (
        [
            ("Szanowni Państwo,", 62, BOLD),
            ("w związku z remontem wejście", 62, REG),
            ("główne jest nieczynne do", 62, REG),
            ("30 listopada. Prosimy korzystać", 62, REG),
            ("z wejścia od ulicy Piwnej.", 62, REG),
            ("Dziękujemy za wyrozumiałość!", 62, REG),
        ],
        (245, 245, 235), (20, 20, 20),
    ),
    "render_ru": (
        [("ВЫХОД В ГОРОД", 120, BOLD), ("к улице Арбат", 90, REG), ("Не загораживать!", 80, REG)],
        (180, 60, 20), (255, 255, 255),
    ),
}
gt = {}
for name, (lines, bg, fg) in signs.items():
    img = on_wall(sign(lines, bg, fg))
    cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    gt[name] = [t for t, _, _ in lines]
base = cv2.imread(str(OUT / "render_de.jpg"))
for k, rot in (("90", cv2.ROTATE_90_CLOCKWISE), ("180", cv2.ROTATE_180), ("270", cv2.ROTATE_90_COUNTERCLOCKWISE)):
    cv2.imwrite(str(OUT / f"render_de_rot{k}.jpg"), cv2.rotate(base, rot), [cv2.IMWRITE_JPEG_QUALITY, 88])
    gt[f"render_de_rot{k}"] = gt["render_de"]
cv2.imwrite(str(OUT / "render_de_persp.jpg"), perspective(base), [cv2.IMWRITE_JPEG_QUALITY, 88])
gt["render_de_persp"] = gt["render_de"]
# skewed paragraph: 15 degree in-plane tilt
para = cv2.imread(str(OUT / "render_pl_paragraph.jpg"))
h, w = para.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
cv2.imwrite(str(OUT / "render_pl_tilt15.jpg"), cv2.warpAffine(para, M, (w, h), borderValue=(140, 140, 140)))
gt["render_pl_tilt15"] = gt["render_pl_paragraph"]
json.dump(gt, open(OUT / "render_gt.json", "w"), ensure_ascii=False, indent=1)
print(sorted(gt))
