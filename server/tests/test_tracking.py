from glass_vision.pipelines.base import Annotation
from glass_vision.tracking import TrackSmoother

W, H = 640, 480
FRAME_MS = 100  # 10 fps


def frame(*dets: tuple[float, str, float]) -> list[Annotation]:
    """dets: (x offset px, label, conf) of a 100x200 box at y=100."""
    return [Annotation((x, 100, x + 100, 300), conf, label) for x, label, conf in dets]


def run(frames: list[list[Annotation]], conf: float | None = 0.35, hold_ms: int = 300) -> list[list]:
    smoother = TrackSmoother("objects")
    out = []
    for i, anns in enumerate(frames):
        smoother.update(anns, W, H, conf, i * FRAME_MS)
        out.append(smoother.views(hold_ms, i * FRAME_MS))
    return out


def test_two_missed_frames_in_ten_do_not_flicker() -> None:
    # The reported case: a person in 8 of 10 frames, missed entirely in 2.
    frames = [frame() if i in (4, 7) else frame((100 + i * 3, "person", 0.8)) for i in range(10)]
    out = run(frames)
    assert all(len(views) == 1 for views in out)
    assert {views[0].id for views in out} == {out[0][0].id}  # one stable identity
    assert out[0][0].id.startswith("objects:")
    assert out[4][0].seen_ms_ago == FRAME_MS and out[5][0].seen_ms_ago == 0


def test_confidence_dip_below_threshold_keeps_track() -> None:
    confs = [0.8, 0.8, 0.8, 0.25, 0.2, 0.8, 0.8]
    out = run([frame((100, "person", c)) for c in confs], hold_ms=0)
    assert all(len(views) == 1 and views[0].seen_ms_ago == 0 for views in out)
    assert out[4][0].conf > 0.4  # EMA, not the raw 0.2


def test_object_gone_disappears_after_hold() -> None:
    out = run([frame((100, "person", 0.9))] * 5 + [frame()] * 6, hold_ms=300)
    assert [len(v) for v in out[5:]] == [1, 1, 1, 0, 0, 0]


def test_one_frame_ghost_is_not_shown() -> None:
    base = frame((100, "person", 0.9))
    frames = [base] * 3 + [base + frame((400, "cup", 0.6))] + [base] * 3
    out = run(frames)
    assert all(len(views) == 1 and views[0].label == "person" for views in out)


def test_label_vote_resists_single_frame_flip() -> None:
    labels = ["cup", "cup", "cup", "cup", "bowl", "cup", "cup"]
    out = run([frame((100, label, 0.7)) for label in labels])
    assert [views[0].label for views in out] == ["cup"] * 7


def test_prefiltered_pipeline_accepts_low_scores_and_carries_payload() -> None:
    # conf=None: the pipeline already filtered (faces, OCR, gestures); low scores must still track.
    anns = [Annotation((100, 100, 200, 300), 0.15, "Alex", detail="0.71", points=[(150.0, 150.0, 1.0)])]
    out = run([anns, anns, anns], conf=None)
    assert all(len(views) == 1 for views in out)
    view = out[-1][0]
    assert (view.label, view.detail) == ("Alex", "0.71")
    assert view.points == [(150 / W, 150 / H, 1.0)]
