"""Same immutable experiment queue and results; reviewed proposal/report helpers."""
from . import common as c

def main():
    c.verify()
    for name in ('RUNTIME_ADDENDUM_001.json','RUNTIME_ADDENDUM_002.json'):
        for p,w in c.read(c.HERE/name)['files'].items():assert c.digest(p)==w
    from .report_runtime_fix_001 import install as report_install
    from .proposal_runtime_002 import install as proposal_install
    report_install();proposal_install()
    from .supervise import main as queue
    queue()

if __name__=='__main__':main()
