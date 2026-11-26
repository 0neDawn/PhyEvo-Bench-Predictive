"""Route the unchanged NN0 CUDA scorer to the separate S namespace."""
from . import common as c


if __name__=='__main__':
    c.install()
    from predictive_tracks.ppr_eta_radius_20261104_o.cache_worker import main
    main()
