"""Fixed eight-observation state anchors; no extra neural computation."""

import numpy as np
from PIL import Image

HISTORY_SCHEDULES = {
    "recent8": tuple(range(12, 20)),
    "anchor2_recent6": (8, 9, 14, 15, 16, 17, 18, 19),
}


def read_history_view(record, schedule):
    indices = HISTORY_SCHEDULES[schedule] + tuple(range(20, 26))
    videos = []
    for directory in (record.possible_path, record.impossible_path):
        paths = {int(p.stem): p for p in directory.glob("*.png")}
        frames = []
        for index in indices:
            with Image.open(paths[index]) as image:
                frames.append(np.asarray(image.convert("RGB")).copy())
        videos.append(np.stack(frames))
    if not np.array_equal(videos[0][:8], videos[1][:8]):
        raise RuntimeError("New history view is not shared; never silently remove this pair")
    return tuple(videos)


class HistoryContextMixin:
    def __init__(self, *args, orientation="identity", **kwargs):
        if orientation != "identity":
            raise ValueError("Context and orientation interventions must not be silently combined")
        self.last_track_audit = None
        super().__init__(*args, **kwargs)

    def score_pair(self, possible_rgb, impossible_rgb):
        if possible_rgb.shape != impossible_rgb.shape or possible_rgb.shape[0] != 14:
            raise ValueError("Expected eight observations and six candidate future frames")
        if not np.array_equal(possible_rgb[:8], impossible_rgb[:8]):
            raise ValueError("History is not shared")
        before = self.prediction_calls
        result = super().score_pair(possible_rgb, impossible_rgb)
        if self.prediction_calls - before != 1:
            raise RuntimeError("Future forecast count changed")
        self.last_track_audit = {"mode": "anchor2_recent6", "flip": False,
                                 "additional_neural_forwards": 0,
                                 "shared_forecasts": 1, "preprocessing_seconds": 0.0,
                                 "history_frames": list(HISTORY_SCHEDULES["anchor2_recent6"])}
        return result
