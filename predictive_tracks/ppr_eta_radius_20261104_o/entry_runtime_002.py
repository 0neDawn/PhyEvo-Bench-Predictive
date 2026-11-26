"""Same experiment queue, with verified legacy result indexing and lineage."""
from . import common as c
def main():
    c.verify()
    for name in ('PATH_RUNTIME_ADDENDUM_001.json','REPORT_RUNTIME_ADDENDUM_002.json'):
        for p,w in c.read(c.HERE/name)['files'].items():assert c.digest(p)==w
    from .path_runtime_001 import install as paths
    from .report_runtime_002 import install as controls
    paths();controls()
    from .supervise import main as queue
    queue()
if __name__=='__main__':main()
