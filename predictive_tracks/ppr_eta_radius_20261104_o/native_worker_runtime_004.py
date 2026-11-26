"""Native worker; observe original LSTM calls and hash all SlotFormer weights."""
from . import common as c


def main():
    from .parameter_runtime_004 import install_worker, verify_addendum
    c.verify()
    verify_addendum()
    install_worker()
    from .native_worker import main as run
    run()


if __name__ == "__main__":
    main()
