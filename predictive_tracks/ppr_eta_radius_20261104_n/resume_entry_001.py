"""Resume original serial queue with a declared reporting-only runtime fix."""
from . import common as c
from .report_runtime_fix_001 import install

def main():
    c.verify()
    for p,w in c.read(c.HERE/'RUNTIME_ADDENDUM_001.json')['files'].items():assert c.digest(p)==w
    report=install()
    # Exercise the fixed cumulative assembly before resuming any GPU worker.
    result=report.assemble();assert result['completed_new_configurations']>=126
    from .supervise import main as queue
    queue()

if __name__=='__main__':main()
