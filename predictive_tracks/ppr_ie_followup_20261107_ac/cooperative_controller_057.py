"""Change only owned scheduling; keep scientific phases and resource limits."""
from pathlib import Path
from . import common as c

def install():
    from . import controller_012 as original
    c.verify_sources('PRIORITY_ENTRY_READY_056.json')
    source=Path(original.__file__).read_text()
    begin=source.index('    def execute(model,phase):\n')
    end=source.index('    try:\n        while not (c.HERE/',begin)
    execute=(c.HERE/'cooperative_execute_057.txt').read_text()
    namespace=dict(vars(original))
    exec(compile(source[:begin]+execute+source[end:],'<AC_cooperative_scheduler057>','exec'),namespace)
    original.main=namespace['main']
