"""Resume the immutable queue with a read-only historical-path adapter."""
from . import common as c
def main():
    c.verify()
    for p,w in c.read(c.HERE/'PATH_RUNTIME_ADDENDUM_001.json')['files'].items():assert c.digest(p)==w
    from .path_runtime_001 import install
    install()
    from .supervise import main as queue
    queue()
if __name__=='__main__':main()
