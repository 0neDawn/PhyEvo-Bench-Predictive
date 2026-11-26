"""Wall-clock cooling only: never read, write, or replace neural inputs."""
import subprocess
import time


class ThermalGuard:
    def __init__(self, record, probe=None, clock=None, sleep=None):
        self.record = record
        self.probe = probe or self.temperature
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.last_probe = -float('inf')

    @staticmethod
    def temperature():
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=5, check=True)
        values = result.stdout.strip().splitlines()
        assert len(values) == 1, values
        return float(values[0])

    def check(self):
        now = self.clock()
        if now - self.last_probe < .5:
            return
        self.last_probe = now
        temperature = self.probe()
        if temperature < 58:
            return
        started = self.clock()
        first = temperature
        maximum = temperature
        while temperature > 50:
            self.sleep(1)
            temperature = self.probe()
            maximum = max(maximum, temperature)
        self.last_probe = self.clock()
        self.record(dict(at=time.time(),action='Pause only this experimental worker between original module calls',
            initial_temperature_c=first,maximum_temperature_c=maximum,
            released_temperature_c=temperature,seconds=self.clock()-started,
            neural_inputs_outputs_order_weights_and_forward_count_unchanged=True))


def install(record):
    import torch
    guard = ThermalGuard(record)
    def before_original_call(module, inputs):
        if torch.cuda.is_initialized():
            guard.check()
        # A pre-hook returning None preserves the original arguments.
    handle = torch.nn.modules.module.register_module_forward_pre_hook(before_original_call)
    return handle
