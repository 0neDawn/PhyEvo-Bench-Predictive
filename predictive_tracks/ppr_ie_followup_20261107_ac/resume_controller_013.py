"""Keep interrupted report attempts; resume original scientific phases unchanged."""
import inspect
import time
from . import common as c


def main():
    c.verify()
    from . import report_010 as report
    original = report.finish
    def finish():
        output = c.HERE/'delivery_20261107_001'
        number = 1
        while output.exists():
            manifest = output/'DELIVERY.json'
            if manifest.exists():
                for p,wanted in c.read(manifest)['files'].items():assert c.digest(p)==wanted,p
                assert c.read(output/'COMPLETENESS_AUDIT.json')['all_original_metrics_recomputed']
                if not (c.HERE/'FINAL_DELIVERY.json').exists():
                    c.save(c.HERE/'FINAL_DELIVERY.json',dict(at=time.time(),directory=str(output),complete=True))
                if not (c.HERE/'FINAL_STATE_001.json').exists():
                    c.save(c.HERE/'FINAL_STATE_001.json',dict(at=time.time(),all_declared_configurations_complete=True,
                        all_lineage_audits_complete=True,every_new_primary_original_full600_validated=True,
                        dino_diagnostic_only=True,directory=str(output),report_metadata_recovered_after_power_cut=True))
                return
            number += 1
            output = c.HERE/f'delivery_20261107_{number:03d}'
        source = inspect.getsource(original)
        token = "out = c.HERE/'delivery_20261107_001'"
        assert source.count(token)==1
        namespace = dict(vars(report),_recovery_output_013=output)
        exec(compile(source.replace(token,'out = _recovery_output_013'),'<AC_report_recovery013>','exec'),namespace)
        namespace['finish']()
    report.finish = finish
    from .controller_012 import main as run
    run()


if __name__ == '__main__':main()
