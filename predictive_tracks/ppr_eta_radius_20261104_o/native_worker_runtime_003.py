"""Original experiment worker with a stricter value audit for native LSTM."""
from . import common as c


def main():
    from .parameter_runtime_003 import install_worker, verify_addendum
    c.verify()
    verify_addendum()
    install_worker()
    from .native_worker import main as run
    run()


if __name__ == "__main__":
    main()
