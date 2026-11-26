"""New original-budget OCVP native-state entry."""
from predictive_tracks.state_tracks_20261023.all_models_20261023.method import CaptureOnlyMixin
from .track_moment_ocvp import Adapter as FrozenInterception
class Adapter(CaptureOnlyMixin, FrozenInterception):
    pass
