"""New G-SWM entry; all original eight draws and alpha weights retained."""
from predictive_tracks.state_tracks_20261023.all_models_20261023.method import CaptureOnlyMixin
from .track_moment import Adapter as FrozenInterception
class Adapter(CaptureOnlyMixin, FrozenInterception):
    pass
