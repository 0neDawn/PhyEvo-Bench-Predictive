"""Same scientific S plan, with the report index initialized."""
from . import common as c

def main():
    from .report_runtime_004 import install
    install()
    from .resume_003 import main as run
    run()

if __name__=='__main__':main()
